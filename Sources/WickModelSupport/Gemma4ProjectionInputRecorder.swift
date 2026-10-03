import Foundation
import MLX

/// One dense projection or one explicitly selected routed expert.
public struct Gemma4ProjectionInputTarget: Hashable {
    public let path: String
    public let expert: Int?

    public init(path: String, expert: Int? = nil) {
        self.path = path
        self.expert = expert
    }
}

/// Actual captured inputs, with total routed/dense coverage reported separately.
public struct Gemma4CapturedProjectionInputs {
    public let target: Gemma4ProjectionInputTarget
    public let inputs: MLXArray
    public let observedPositions: Int
}

/// Bounded deterministic prefix capture on the caller's serialized MLX worker.
/// Create a new recorder for each layer. It never forms or retains covariances.
/// The byte budget reserves room for retained arrays plus final concatenation;
/// it is not a bound on the source model, backend scratch or process footprint.
public final class Gemma4ProjectionInputRecorder: LagunaRoutedActivationObserver {
    private struct Entry {
        var arrays = [MLXArray]()
        var observed = 0
        var captured = 0
        var width: Int?
        var dtype: DType?
    }

    private let targets: Set<Gemma4ProjectionInputTarget>
    private let maximumPositions: Int
    private let maximumRetainedBytes: Int
    private var retainedBytes = 0
    private var entries = [Gemma4ProjectionInputTarget: Entry]()
    private var failure: Error?

    public init(
        targets: [Gemma4ProjectionInputTarget], maximumPositions: Int,
        maximumRetainedBytes: Int
    ) throws {
        guard !targets.isEmpty, Set(targets).count == targets.count,
            targets.allSatisfy({ !$0.path.isEmpty && ($0.expert ?? 0) >= 0 }),
            maximumPositions > 0, maximumPositions <= Int(Int32.max), maximumRetainedBytes > 0
        else {
            throw Gemma4CalibrationError.invalidInput("invalid projection capture targets or limits")
        }
        self.targets = Set(targets)
        self.maximumPositions = maximumPositions
        self.maximumRetainedBytes = maximumRetainedBytes
    }

    /// Records only explicitly selected dense projections; other paths are ignored.
    public func observeDense(path: String, input: MLXArray) {
        let target = Gemma4ProjectionInputTarget(path: path)
        guard targets.contains(target), failure == nil else { return }
        do {
            guard input.ndim > 0, input.dim(-1) > 0, input.size > 0 else {
                throw Gemma4CalibrationError.invalidInput("invalid dense capture geometry")
            }
            let width = input.dim(-1)
            let positions = input.size / width
            let remaining = maximumPositions - (entries[target]?.captured ?? 0)
            try append(
                target: target, input: input.reshaped(positions, width),
                positions: Array(0..<min(positions, remaining)), observed: positions)
        } catch {
            failure = error
        }
    }

    /// Conditions gate/up and post-activation down inputs on actual selected IDs.
    public func observeRoutedProjection(
        path: String, input: MLXArray, indices: MLXArray, expertCount: Int
    ) {
        let selected = targets.filter { $0.path == path }
        guard !selected.isEmpty, failure == nil else { return }
        do {
            guard expertCount > 0, input.ndim >= 2, input.dim(-2) == 1, input.dim(-1) > 0,
                indices.size > 0, indices.dtype == .int32 || indices.dtype == .uint32,
                selected.allSatisfy({ $0.expert != nil && $0.expert! < expertCount })
            else {
                throw Gemma4CalibrationError.invalidInput("invalid routed capture geometry or expert target")
            }
            let width = input.dim(-1)
            let shape = indices.shape + [1, width]
            guard input.ndim <= shape.count else {
                throw Gemma4CalibrationError.invalidInput("routed input rank exceeds selected expert shape")
            }
            let padded = Array(repeating: 1, count: shape.count - input.ndim) + input.shape
            guard zip(padded, shape).allSatisfy({ $0.0 == 1 || $0.0 == $0.1 }) else {
                throw Gemma4CalibrationError.invalidInput("routed input cannot broadcast to selected experts")
            }
            let ids = indices.flattened().asType(.int32)
            try MLX.checkedEval(ids)
            let values = ids.asArray(Int32.self)
            guard values.allSatisfy({ $0 >= 0 && $0 < expertCount }) else {
                throw Gemma4CalibrationError.invalidInput("out-of-range routed capture expert ID")
            }
            let routed = MLX.broadcast(input, to: shape).reshaped(indices.size, width)
            for target in selected {
                let positions = values.enumerated().compactMap { index, value in
                    Int(value) == target.expert! ? index : nil
                }
                let remaining = maximumPositions - (entries[target]?.captured ?? 0)
                try append(
                    target: target, input: routed, positions: Array(positions.prefix(remaining)),
                    observed: positions.count)
            }
        } catch {
            failure = error
        }
    }

    /// Surfaces deferred observer errors before advancing to the next segment.
    public func evaluatePending() throws {
        if let failure { throw failure }
        try MLX.checkedEval(entries.values.flatMap(\.arrays))
    }

    /// Returns observed targets only. Zero-coverage targets fail explicitly so the
    /// caller must record a low-coverage fallback rather than invent a covariance.
    public func finalize() throws -> [Gemma4CapturedProjectionInputs] {
        try evaluatePending()
        return try targets.sorted {
            $0.path == $1.path ? ($0.expert ?? -1) < ($1.expert ?? -1) : $0.path < $1.path
        }.map { target in
            guard let entry = entries[target], entry.captured > 0 else {
                throw Gemma4CalibrationError.invalidInput("selected projection has no captured positions")
            }
            let input = MLX.concatenated(entry.arrays, axis: 0)
            try MLX.checkedEval(input)
            return Gemma4CapturedProjectionInputs(
                target: target, inputs: input, observedPositions: entry.observed)
        }
    }

    private func append(
        target: Gemma4ProjectionInputTarget, input: MLXArray, positions: [Int], observed: Int
    ) throws {
        guard input.ndim == 2, input.dim(1) > 0,
            input.dtype == .bfloat16 || input.dtype == .float16 || input.dtype == .float32
        else { throw Gemma4CalibrationError.invalidInput("capture requires floating point input matrices") }
        var entry = entries[target] ?? Entry()
        guard entry.width == nil || entry.width == input.dim(1),
            entry.dtype == nil || entry.dtype == input.dtype
        else { throw Gemma4CalibrationError.invalidInput("projection capture geometry changed") }
        let (count, overflow) = entry.observed.addingReportingOverflow(observed)
        guard !overflow else { throw Gemma4CalibrationError.invalidInput("capture position count overflow") }
        entry.observed = count
        entry.width = input.dim(1)
        entry.dtype = input.dtype
        if !positions.isEmpty {
            let (elements, elementOverflow) = positions.count.multipliedReportingOverflow(by: input.dim(1))
            let bytesPerElement = input.dtype == .float32 ? 4 : 2
            let (bytes, byteOverflow) = elements.multipliedReportingOverflow(by: bytesPerElement)
            let (total, totalOverflow) = retainedBytes.addingReportingOverflow(bytes)
            guard !elementOverflow, !byteOverflow, !totalOverflow, total <= maximumRetainedBytes / 2 else {
                throw Gemma4CalibrationError.invalidInput("projection capture exceeds retained/concatenation budget")
            }
            let captured = input[MLXArray(positions)]
            try MLX.checkedEval(captured)
            guard MLX.all(MLX.isFinite(captured)).item(Bool.self) else {
                throw Gemma4CalibrationError.invalidInput("non-finite captured projection input")
            }
            entry.arrays.append(captured)
            entry.captured += positions.count
            retainedBytes = total
        }
        entries[target] = entry
    }
}

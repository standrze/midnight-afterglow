import Foundation
import MLX

public struct Gemma4DenseActivationStatistics {
    public let path: String
    public let secondMoments: MLXArray
    public let positionCount: Int
}

public struct Gemma4CollectedActivations {
    public let dense: [Gemma4DenseActivationStatistics]
    public let experts: [LagunaExpertActivationStatistics]
    public let observedTokenCount: Int
    public let segmentTokenCounts: [Int]
}

/// Collects from independent, already-tokenized corpus segments. Tokenization,
/// source/corpus identity, fit/dev separation and output publication belong to
/// the caller. This collector never changes or writes the source checkpoint.
public enum Gemma4ActivationCollector {
    public static func collect(
        source: URL, tokenSegments: [[Int]], spoolParent: URL,
        minimumExpertPositions: Int = 32,
        projectionInputRecorder: ((Int) throws -> Gemma4ProjectionInputRecorder?)? = nil,
        consumeProjectionInputs: ((Int, [Gemma4CapturedProjectionInputs]) throws -> Void)? = nil
    ) throws -> Gemma4CollectedActivations {
        guard (projectionInputRecorder == nil) == (consumeProjectionInputs == nil) else {
            throw Gemma4CalibrationError.invalidInput("projection capture requires both factory and consumer")
        }
        let source = source.standardizedFileURL.resolvingSymlinksInPath()
        let parent = spoolParent.standardizedFileURL.resolvingSymlinksInPath()
        guard parent != source, !parent.path.hasPrefix(source.path + "/") else {
            throw Gemma4CalibrationError.invalidInput("activation spool must be outside the source checkpoint")
        }
        let configuration = try Gemma4CalibrationConfiguration(
            data: Data(contentsOf: source.appendingPathComponent("config.json")))
        guard !tokenSegments.isEmpty, tokenSegments.allSatisfy({ !$0.isEmpty }), minimumExpertPositions > 0 else {
            throw Gemma4CalibrationError.invalidInput("require nonempty token segments and positive expert coverage")
        }
        let reader = try SelectiveSafetensorsReader(directory: source)
        let keys = reader.keys
        let roots = ["model.language_model", "language_model.model", "language_model", "model"]
        func sourceKey(_ suffix: String) throws -> String {
            let candidates = roots.map { $0 + "." + suffix }.filter { keys.contains($0) }
            guard candidates.count == 1 else {
                throw Gemma4CalibrationError.invalidWeights("missing or ambiguous source tensor: \(suffix)")
            }
            return candidates[0]
        }
        let embeddingKey = try sourceKey("embed_tokens.weight")
        let embeddingDescription = try reader.description(for: embeddingKey)
        guard embeddingDescription.dtype == .bfloat16,
            embeddingDescription.shape.count == 2,
            embeddingDescription.shape[0] > 0,
            embeddingDescription.shape[1] == configuration.hiddenSize
        else { throw Gemma4CalibrationError.invalidWeights("expected BF16 vocabulary embedding") }
        var total = 0
        for segment in tokenSegments {
            guard segment.allSatisfy({ $0 >= 0 && $0 < embeddingDescription.shape[0] }) else {
                throw Gemma4CalibrationError.invalidInput("token ID outside source vocabulary")
            }
            let (next, overflow) = total.addingReportingOverflow(segment.count)
            guard !overflow, next <= Int(Int32.max) else {
                throw Gemma4CalibrationError.invalidInput("token position count overflow")
            }
            total = next
        }
        let (elements, elementOverflow) = total.multipliedReportingOverflow(by: configuration.hiddenSize)
        let (stageBytes, byteOverflow) = elements.multipliedReportingOverflow(by: 4)
        guard !elementOverflow, !byteOverflow, stageBytes <= Int.max - 1_048_576 else {
            throw Gemma4CalibrationError.invalidInput("activation spool size overflow")
        }
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        if let available = try parent.resourceValues(forKeys: [.volumeAvailableCapacityKey]).volumeAvailableCapacity,
            available < stageBytes + 1_048_576
        {
            throw Gemma4CalibrationError.invalidInput("insufficient disk space for two BF16 activation stages")
        }
        let spool = parent.appendingPathComponent("gemma4-activations-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: spool, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: spool) }
        func segmentURL(_ stage: URL, _ index: Int) -> URL {
            stage.appendingPathComponent(String(format: "segment-%06d.safetensors", index))
        }
        var stage = spool.appendingPathComponent("stage-0", isDirectory: true)
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
        try pooled {
            let embedding = try reader.read(embeddingKey)
            for (index, segment) in tokenSegments.enumerated() {
                try pooled {
                    let tokens = MLXArray(segment).reshaped(1, segment.count)
                    // Matches Gemma4TextModelInner's BF16 embedding scale.
                    let hidden = embedding[tokens] * Float(configuration.hiddenSize).squareRoot()
                    try MLX.checkedEval(hidden)
                    try MLX.save(arrays: ["hidden": hidden], url: segmentURL(stage, index))
                }
            }
        }
        Memory.clearCache()
        var denseResults = [Gemma4DenseActivationStatistics]()
        var expertResults = [LagunaExpertActivationStatistics]()
        for layerIndex in 0..<configuration.layerCount {
            let nextStage = spool.appendingPathComponent("stage-\(layerIndex + 1)", isDirectory: true)
            try FileManager.default.createDirectory(at: nextStage, withIntermediateDirectories: false)
            let inputRecorder = try projectionInputRecorder?(layerIndex)
            let collected = try pooled {
                let prefixes = roots.map { $0 + ".layers.\(layerIndex)." }
                let layerKeys = keys.filter { key in prefixes.contains { key.hasPrefix($0) } }
                let dense = Gemma4DenseMomentRecorder()
                let routed = try LagunaRoutedActivationRecorder(minimumExpertPositions: minimumExpertPositions)
                let block = try Gemma4CalibrationBlock(
                    configuration: configuration, layerIndex: layerIndex,
                    sourceWeights: reader.read(keys: layerKeys),
                    observeDense: { path, input in
                        dense.observe(path: path, input: input)
                        inputRecorder?.observeDense(path: path, input: input)
                    }, observeRouted: Gemma4CombinedRoutedObserver(moments: routed, inputs: inputRecorder))
                for (index, segment) in tokenSegments.enumerated() {
                    try pooled {
                        let hidden = try loadHidden(stage, index: index, tokens: segment.count)
                        let next = try block(hidden)
                        try MLX.checkedEval(next)
                        try dense.evaluatePending()
                        if !block.routedProjectionPaths.isEmpty { try routed.evaluatePending() }
                        try inputRecorder?.evaluatePending()
                        try MLX.save(arrays: ["hidden": next], url: segmentURL(nextStage, index))
                    }
                    Memory.clearCache()
                }
                let denseValues = try dense.finalize(widths: block.denseProjectionWidths, positions: total)
                let experts = try block.routedProjectionPaths.isEmpty ? [] : routed.finalize()
                guard Set(experts.map(\.path)) == block.routedProjectionPaths else {
                    throw Gemma4CalibrationError.invalidInput("incomplete routed projection coverage")
                }
                if let inputRecorder {
                    try consumeProjectionInputs?(layerIndex, inputRecorder.finalize())
                }
                return (denseValues, experts)
            }
            denseResults.append(contentsOf: collected.0)
            expertResults.append(contentsOf: collected.1)
            try FileManager.default.removeItem(at: stage)
            stage = nextStage
            Memory.clearCache()
        }
        // The final normalized hidden state is the head input, including a tied
        // embedding head. No vocabulary-sized logits need to be materialized.
        let head = try pooled {
            let norm = try reader.read(sourceKey("norm.weight"))
            guard norm.dtype == .bfloat16, norm.shape == [configuration.hiddenSize] else {
                throw Gemma4CalibrationError.invalidWeights("expected BF16 final norm")
            }
            let path = configuration.moduleRoot == "model" ? "lm_head" : "language_model.lm_head"
            let recorder = Gemma4DenseMomentRecorder()
            for (index, segment) in tokenSegments.enumerated() {
                try pooled {
                    let hidden = try loadHidden(stage, index: index, tokens: segment.count)
                    let normalized = MLXFast.rmsNorm(hidden, weight: norm, eps: 0.000001)
                    recorder.observe(path: path, input: normalized)
                    try recorder.evaluatePending()
                }
                Memory.clearCache()
            }
            return try recorder.finalize(widths: [path: configuration.hiddenSize], positions: total)
        }
        denseResults.append(contentsOf: head)
        return Gemma4CollectedActivations(
            dense: denseResults, experts: expertResults, observedTokenCount: total,
            segmentTokenCounts: tokenSegments.map(\.count))

        func loadHidden(_ stage: URL, index: Int, tokens: Int) throws -> MLXArray {
            let arrays = try MLX.loadArrays(url: segmentURL(stage, index), stream: .cpu)
            guard arrays.count == 1, let hidden = arrays["hidden"],
                hidden.dtype == .bfloat16, hidden.shape == [1, tokens, configuration.hiddenSize]
            else { throw Gemma4CalibrationError.invalidInput("activation spool shape/dtype mismatch") }
            return hidden
        }
    }

    private static func pooled<T>(_ body: () throws -> T) rethrows -> T {
        #if canImport(ObjectiveC)
            return try autoreleasepool(invoking: body)
        #else
            return try body()
        #endif
    }
}

private final class Gemma4DenseMomentRecorder {
    private struct Entry {
        var sums: MLXArray
        var count: Int
    }
    private var entries = [String: Entry]()
    private var failure: String?

    func observe(path: String, input: MLXArray) {
        guard failure == nil else { return }
        guard input.ndim > 0, input.dim(-1) > 0, input.size > 0 else {
            failure = "invalid dense activation geometry for \(path)"
            return
        }
        let width = input.dim(-1)
        let positions = input.size / width
        var entry = entries[path] ?? Entry(sums: MLXArray.zeros([width], type: Float.self), count: 0)
        let (count, overflow) = entry.count.addingReportingOverflow(positions)
        guard !overflow, entry.sums.shape == [width] else {
            failure = "changed dense geometry or position overflow for \(path)"
            return
        }
        entry.sums = entry.sums + MLX.sum(MLX.square(input.reshaped(positions, width).asType(.float32)), axis: 0)
        entry.count = count
        entries[path] = entry
    }

    func evaluatePending() throws {
        if let failure { throw Gemma4CalibrationError.invalidInput(failure) }
        try MLX.checkedEval(entries.values.map(\.sums))
    }

    func finalize(widths: [String: Int], positions: Int) throws -> [Gemma4DenseActivationStatistics] {
        try evaluatePending()
        guard Set(entries.keys) == Set(widths.keys) else {
            throw Gemma4CalibrationError.invalidInput("incomplete dense projection coverage")
        }
        return try widths.keys.sorted().map { path in
            let entry = entries[path]!
            guard entry.count == positions, entry.sums.shape == [widths[path]!] else {
                throw Gemma4CalibrationError.invalidInput("dense position coverage mismatch for \(path)")
            }
            let moments = entry.sums / Float(positions)
            try MLX.checkedEval(moments)
            guard moments.asArray(Float.self).allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                throw Gemma4CalibrationError.invalidInput("non-finite dense moments for \(path)")
            }
            return Gemma4DenseActivationStatistics(path: path, secondMoments: moments, positionCount: positions)
        }
    }
}

/// Forwards the same unmodified routed inputs to moments and optional capture.
private final class Gemma4CombinedRoutedObserver: LagunaRoutedActivationObserver {
    private let moments: LagunaRoutedActivationRecorder
    private let inputs: Gemma4ProjectionInputRecorder?

    init(moments: LagunaRoutedActivationRecorder, inputs: Gemma4ProjectionInputRecorder?) {
        self.moments = moments
        self.inputs = inputs
    }

    func observeRoutedProjection(path: String, input: MLXArray, indices: MLXArray, expertCount: Int) {
        moments.observeRoutedProjection(path: path, input: input, indices: indices, expertCount: expertCount)
        inputs?.observeRoutedProjection(path: path, input: input, indices: indices, expertCount: expertCount)
    }
}

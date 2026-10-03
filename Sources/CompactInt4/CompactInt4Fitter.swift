import Foundation

extension CompactInt4Matrix {
    /// Bounded alternating fit of codes, shared offsets, discrete exponents and row gain.
    /// Uses weight reconstruction error only; activation-aware calibration is a separate step.
    public static func fit(
        _ weights: [Float], rows: Int, columns: Int, iterations: Int = 6, importance: [Float]? = nil,
        scaleEncoding: ScaleEncoding = .e8m0
    ) throws -> Self {
        guard scaleEncoding != .e3m4 else {
            throw CompactInt4Error.invalid("Signed E3M4 uses the exploratory covariance fitter; load its artifact")
        }
        if importance != nil || scaleEncoding != .e8m0 {
            return try fitCalibrated(
                weights, rows: rows, columns: columns, iterations: iterations, importance: importance,
                encoding: scaleEncoding)
        }
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard rows > 0, columns > 0, columns % 64 == 0, !overflow, weights.count == count,
            weights.allSatisfy(\.isFinite), (1...16).contains(iterations)
        else { throw CompactInt4Error.invalid("Invalid fitting matrix or iteration bound") }
        var packed = [UInt8](repeating: 0, count: count / 2)
        var exponents = [UInt8](repeating: 127, count: count / 32)
        var offsets = [UInt16](repeating: 0, count: count / 64)
        var gains = [UInt16](repeating: bf16(1), count: rows)
        for row in 0..<rows {
            let source = weights[(row * columns)..<((row + 1) * columns)].map(Double.init)
            let peak = source.map(abs).max() ?? 0
            // Seed across one octave: row gain removes much of power-of-two scale rigidity.
            var best: RowFit?
            for seed in [0.75, 0.875, 1.0, 1.125, 1.25, 1.5] {
                let initial = peak == 0 ? 1 : peak / 7 * seed
                let candidate = fitRow(source, gain: initial, iterations: iterations)
                if candidate.loss.isFinite && (best == nil || candidate.loss < best!.loss) {
                    best = candidate
                }
            }
            guard let best else { throw CompactInt4Error.invalid("Weights exceed representable fitting range") }
            gains[row] = best.gain
            for k in 0..<columns {
                let index = row * columns + k
                packed[index / 2] |= UInt8(best.codes[k] & 15) << ((index % 2) * 4)
            }
            exponents.replaceSubrange((row * columns / 32)..<((row + 1) * columns / 32), with: best.scaleBytes)
            offsets.replaceSubrange((row * columns / 64)..<((row + 1) * columns / 64), with: best.offsets)
        }
        return try Self(
            rows: rows, columns: columns, codes: packed, scaleBytes: exponents, offsets: offsets, gains: gains)
    }

    private struct RowFit {
        var gain: UInt16
        var codes: [Int]
        var scaleBytes: [UInt8]
        var offsets: [UInt16]
        var loss = Double.infinity
    }

    private static func fitRow(_ source: [Double], gain initialGain: Double, iterations: Int) -> RowFit {
        let width = source.count
        var state = RowFit(
            gain: bf16(Float(initialGain)), codes: .init(repeating: 0, count: width),
            scaleBytes: .init(repeating: 127, count: width / 32),
            offsets: .init(repeating: 0, count: width / 64))
        var best = state
        guard float(state.gain).isFinite, float(state.gain) > 0 else { return best }
        for iteration in 0..<iterations {
            let gain = Double(float(state.gain))
            for block in 0..<(width / 64) {
                let start = block * 64
                if iteration == 0 {
                    let mean = source[start..<(start + 64)].reduce(0, +) / (64 * gain)
                    state.offsets[block] = bf16(Float(mean))
                }
                let bias = Double(float(state.offsets[block]))
                for half in 0..<2 {
                    let group = start + half * 32
                    let normalized = source[group..<(group + 32)].map { $0 / gain - bias }
                    let magnitude = normalized.map(abs).max() ?? 0
                    let center = magnitude > 0 && magnitude.isFinite ? Int(log2(magnitude / 7).rounded()) : 0
                    var groupLoss = Double.infinity
                    var groupCodes = [Int](repeating: 0, count: 32)
                    var selectedExponent: UInt8 = 127
                    for exponent in max(-127, center - 2)...max(-127, min(127, center + 2)) {
                        let step = pow(2.0, Double(exponent))
                        var loss: Double = 0
                        var codes = [Int]()
                        for value in normalized {
                            let code = Int(min(7, max(-8, (value / step).rounded(.toNearestOrEven))))
                            codes.append(code)
                            loss += pow(value - Double(code) * step, 2)
                        }
                        if loss < groupLoss {
                            groupLoss = loss
                            groupCodes = codes
                            selectedExponent = UInt8(exponent + 127)
                        }
                    }
                    state.codes.replaceSubrange(group..<(group + 32), with: groupCodes)
                    state.scaleBytes[group / 32] = selectedExponent
                }
                // Exact least-squares shared offset for fixed codes/exponents, then BF16 round trip.
                let meanResidual =
                    (start..<(start + 64)).reduce(0.0) { sum, k in
                        sum + source[k] / gain - Double(state.codes[k]) * Double(scale(state.scaleBytes[k / 32]))
                    } / 64
                state.offsets[block] = bf16(Float(meanResidual))
            }
            var numerator: Double = 0
            var denominator: Double = 0
            for k in 0..<width {
                let normalized =
                    Double(state.codes[k]) * Double(scale(state.scaleBytes[k / 32]))
                    + Double(float(state.offsets[k / 64]))
                numerator += source[k] * normalized
                denominator += normalized * normalized
            }
            let fittedGain = bf16(Float(denominator > 0 ? numerator / denominator : 1))
            if float(fittedGain).isFinite && float(fittedGain) > 0 { state.gain = fittedGain }
            state.loss = (0..<width).reduce(0) { sum, k in
                let reconstructed =
                    (Double(state.codes[k]) * Double(scale(state.scaleBytes[k / 32]))
                        + Double(float(state.offsets[k / 64]))) * Double(float(state.gain))
                return sum + pow(source[k] - reconstructed, 2)
            }
            if state.loss < best.loss { best = state }
        }
        return best
    }
}

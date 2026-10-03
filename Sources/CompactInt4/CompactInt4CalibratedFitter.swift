import Foundation

extension CompactInt4Matrix {
    public enum ScaleEncoding: String, Sendable {
        case e8m0
        /// Positive normalized scale: (1 + mantissa/16) * 2^(exponent - 7).
        case e4m4
        /// Signed normalized scale: sign * (1 + mantissa/16) * 2^(exponent - 3).
        case e3m4
    }

    static func fitCalibrated(
        _ weights: [Float], rows: Int, columns: Int, iterations: Int,
        importance: [Float]?, encoding: ScaleEncoding
    ) throws -> Self {
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard rows > 0, columns > 0, columns % 64 == 0, !overflow, weights.count == count,
            weights.allSatisfy(\.isFinite), (1...16).contains(iterations)
        else { throw CompactInt4Error.invalid("Invalid calibrated fitting geometry") }
        let importance = importance ?? [Float](repeating: 1, count: columns)
        guard importance.count == columns, importance.allSatisfy({ $0.isFinite && $0 >= 0 }),
            importance.contains(where: { $0 > 0 })
        else { throw CompactInt4Error.invalid("Importance must contain finite nonnegative channel moments") }
        var packed = [UInt8](repeating: 0, count: count / 2)
        var scaleBytes = [UInt8](repeating: 0, count: count / 32)
        var offsets = [UInt16](repeating: 0, count: count / 64)
        var gains = [UInt16](repeating: 0, count: rows)
        for row in 0..<rows {
            let source = weights[(row * columns)..<((row + 1) * columns)].map(Double.init)
            let peak = source.map(abs).max() ?? 0
            var bestLoss = Double.infinity
            for seed in [0.875, 1.0, 1.125] {
                let gainBits = bf16(Float(peak > 0 ? peak / (7 * 128) * seed : 1))
                let gain = Double(float(gainBits))
                guard gain.isFinite, gain > 0 else { continue }
                var groups = [CalibratedGroup]()
                var loss: Double = 0
                for start in stride(from: 0, to: columns, by: 64) {
                    let y = source[start..<(start + 64)].map { $0 / gain }
                    let moment = importance[start..<(start + 64)].map { max(Double($0), 1e-9) }
                    let group = fitGroup(y, moment: moment, iterations: iterations, encoding: encoding)
                    groups.append(group)
                    loss += group.loss * gain * gain
                }
                guard loss < bestLoss else { continue }
                bestLoss = loss
                gains[row] = gainBits
                for (groupIndex, group) in groups.enumerated() {
                    offsets[row * columns / 64 + groupIndex] = group.bias
                    for half in 0..<2 { scaleBytes[row * columns / 32 + groupIndex * 2 + half] = group.scales[half] }
                    for k in 0..<64 {
                        let index = row * columns + groupIndex * 64 + k
                        let shift = (index % 2) * 4
                        packed[index / 2] = (packed[index / 2] & ~(15 << shift)) | (UInt8(group.codes[k] & 15) << shift)
                    }
                }
            }
            guard bestLoss.isFinite else { throw CompactInt4Error.invalid("No representable calibrated row") }
        }
        return try Self(
            rows: rows, columns: columns, codes: packed, scaleBytes: scaleBytes,
            offsets: offsets, gains: gains, scaleEncoding: encoding)
    }

    private struct CalibratedGroup {
        var scales: [UInt8]
        var bias: UInt16
        var codes: [Int]
        var loss: Double = .infinity
    }

    private static func fitGroup(
        _ y: [Double], moment: [Double], iterations: Int, encoding: ScaleEncoding
    ) -> CalibratedGroup {
        let mass = moment.reduce(0, +)
        let weightedTarget = zip(y, moment).reduce(0.0) { $0 + $1.0 * $1.1 }
        let initialBias = bf16(Float(weightedTarget / mass))
        let centered = y.map { $0 - Double(float(initialBias)) }
        let initialScales = (0..<2).map { half in
            let block = centered[(half * 32)..<((half + 1) * 32)]
            let radius = max((block.max() ?? 0) / 7, -(block.min() ?? 0) / 8)
            return scaleByte(radius, encoding: encoding)
        }
        var state = CalibratedGroup(scales: initialScales, bias: initialBias, codes: .init(repeating: 0, count: 64))
        var best = state
        for _ in 0..<iterations {
            let bias = Double(float(state.bias))
            let slopes = state.scales.map { Double(scale($0, encoding: encoding)) }
            var aa = [Double](repeating: 0, count: 2)
            var cc = aa
            var dd = aa
            state.loss = 0
            for k in 0..<64 {
                let half = k / 32
                let code = Int(min(7, max(-8, ((y[k] - bias) / slopes[half]).rounded(.toNearestOrEven))))
                state.codes[k] = code
                let q = Double(code)
                let error = y[k] - (q * slopes[half] + bias)
                state.loss += error * error * moment[k]
                aa[half] += moment[k] * q * q
                cc[half] += moment[k] * q
                dd[half] += moment[k] * q * y[k]
            }
            if state.loss < best.loss { best = state }
            var numerator = weightedTarget
            var denominator = mass
            for half in 0..<2 where aa[half] > 1e-12 {
                numerator -= cc[half] * dd[half] / aa[half]
                denominator -= cc[half] * cc[half] / aa[half]
            }
            let fittedBias = denominator > 1e-12 ? numerator / denominator : bias
            guard fittedBias.isFinite else { break }
            state.bias = bf16(Float(fittedBias))
            for half in 0..<2 where aa[half] > 1e-12 {
                let fittedScale = (dd[half] - fittedBias * cc[half]) / aa[half]
                if fittedScale.isFinite && fittedScale > 0 {
                    state.scales[half] = scaleByte(fittedScale, encoding: encoding)
                }
            }
        }
        return best
    }
}

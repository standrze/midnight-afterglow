import Foundation

public enum CompactInt4Error: Error {
    case invalid(String)
}

/// Experimental row-major 4-bit codes / byte-scale-G32 / BF16-offset-G64 matrix.
/// Arithmetic is defined by the stored metadata, with FP32 accumulation.
public struct CompactInt4Matrix: Sendable {
    public enum CodeEncoding: String, Sendable {
        case signedInt4
        case normal16
    }

    /// Fixed FP16 midpoint-normal codebook. Its bit patterns are part of AGQ4N001.
    public static let normal16Bits: [UInt16] = [
        49011, 48454, 48138, 47670, 47266, 46704, 45975, 44293, 11525, 13207, 13936, 14498, 14902, 15370, 15686, 16243,
    ]
    public let codeEncoding: CodeEncoding
    public let scaleEncoding: ScaleEncoding
    public let rows: Int
    public let columns: Int
    public let codes: [UInt8]
    public let scaleBytes: [UInt8]
    public let offsets: [UInt16]
    public let gains: [UInt16]

    public init(
        rows: Int, columns: Int, codes: [UInt8], scaleBytes: [UInt8], offsets: [UInt16], gains: [UInt16],
        scaleEncoding: ScaleEncoding = .e8m0, codeEncoding: CodeEncoding = .signedInt4
    ) throws {
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard rows > 0, columns > 0, columns % 64 == 0, !overflow,
            codes.count == count / 2, scaleBytes.count == count / 32,
            codeEncoding != .normal16 || scaleEncoding == .e4m4,
            offsets.count == count / 64, gains.count == rows,
            scaleEncoding != .e8m0 || scaleBytes.allSatisfy({ $0 != 255 }),
            offsets.allSatisfy({ Self.float($0).isFinite }),
            gains.allSatisfy({ Self.float($0).isFinite && Self.float($0) > 0 })
        else { throw CompactInt4Error.invalid("Invalid compact INT4 geometry or metadata") }
        self.scaleEncoding = scaleEncoding
        self.codeEncoding = codeEncoding
        self.rows = rows
        self.columns = columns
        self.codes = codes
        self.scaleBytes = scaleBytes
        self.offsets = offsets
        self.gains = gains
        for index in 0..<count where !value(at: index).isFinite {
            throw CompactInt4Error.invalid("Metadata produces non-finite reconstructed weights")
        }
    }

    public var payloadBytes: Int {
        codes.count + scaleBytes.count + offsets.count * 2 + gains.count * 2
    }

    public var bitsPerWeight: Double {
        Double(payloadBytes) * 8 / Double(rows * columns)
    }

    public func reconstructed() -> [Float] {
        (0..<(rows * columns)).map { value(at: $0) }
    }

    /// Computes X Wᵀ directly from the packed payload without expanding W.
    public func multiply(_ input: [Float], batch: Int) throws -> [Float] {
        let (inputCount, inputOverflow) = batch.multipliedReportingOverflow(by: columns)
        let (outputCount, outputOverflow) = batch.multipliedReportingOverflow(by: rows)
        guard batch > 0, !inputOverflow, !outputOverflow, input.count == inputCount,
            input.allSatisfy(\.isFinite)
        else { throw CompactInt4Error.invalid("Invalid input matrix") }
        var result = [Float](repeating: 0, count: outputCount)
        for sample in 0..<batch {
            for row in 0..<rows {
                var total: Float = 0
                for block in 0..<(columns / 64) {
                    var inputSum: Float = 0
                    for half in 0..<2 {
                        var dot: Float = 0
                        let start = block * 64 + half * 32
                        for k in start..<(start + 32) {
                            let x = input[sample * columns + k]
                            dot += x * codeValue(at: row * columns + k)
                            inputSum += x
                        }
                        total += dot * Self.scale(scaleBytes[row * columns / 32 + start / 32], encoding: scaleEncoding)
                    }
                    total += inputSum * Self.float(offsets[row * columns / 64 + block])
                }
                result[sample * rows + row] = total * Self.float(gains[row])
            }
        }
        return result
    }

    func code(at index: Int) -> Int {
        let nibble = Int((codes[index / 2] >> ((index % 2) * 4)) & 15)
        return nibble >= 8 ? nibble - 16 : nibble
    }

    func codeValue(at index: Int) -> Float {
        if codeEncoding == .normal16 {
            let nibble = Int((codes[index / 2] >> ((index % 2) * 4)) & 15)
            return Float(Float16(bitPattern: Self.normal16Bits[nibble]))
        }
        return Float(code(at: index))
    }

    func value(at index: Int) -> Float {
        let normalized =
            codeValue(at: index) * Self.scale(scaleBytes[index / 32], encoding: scaleEncoding)
            + Self.float(offsets[index / 64])
        return normalized * Self.float(gains[index / columns])
    }

    static func scale(_ byte: UInt8, encoding: ScaleEncoding = .e8m0) -> Float {
        if encoding == .e3m4 {
            let sign = UInt32(byte & 128) << 24
            let exponent = (UInt32((byte >> 4) & 7) + 124) << 23
            return Float(bitPattern: sign | exponent | UInt32(byte & 15) << 19)
        }
        if encoding == .e4m4 {
            return Float(bitPattern: (UInt32(byte >> 4) + 120) << 23 | UInt32(byte & 15) << 19)
        }
        return Float(bitPattern: byte == 0 ? 0x0040_0000 : UInt32(byte) << 23)
    }

    static func scaleByte(_ value: Double, encoding: ScaleEncoding) -> UInt8 {
        if encoding == .e3m4 {
            let magnitude = min(31, max(0.125, abs(value)))
            let exponent = Int(floor(log2(magnitude)))
            let mantissa = Int(((magnitude / pow(2, Double(exponent)) - 1) * 16).rounded(.toNearestOrEven))
            let positiveByte = UInt8(min(127, max(0, (exponent + 3) * 16 + mantissa)))
            return positiveByte | (value < 0 ? 128 : 0)
        }
        let positive = max(value, Double.leastNormalMagnitude)
        if encoding == .e8m0 {
            return UInt8(min(254, max(0, Int(log2(positive).rounded(.toNearestOrEven)) + 127)))
        }
        let clipped = min(496, max(1.0 / 128, positive))
        let exponent = Int(floor(log2(clipped)))
        let mantissa = Int(((clipped / pow(2, Double(exponent)) - 1) * 16).rounded(.toNearestOrEven))
        return UInt8(min(255, max(0, (exponent + 7) * 16 + mantissa)))
    }

    static func float(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    static func bf16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        return UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
    }
}

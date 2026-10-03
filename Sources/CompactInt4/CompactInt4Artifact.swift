import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

extension CompactInt4Matrix {
    public var artifactFormat: String {
        if codeEncoding == .normal16 { return "AGQ4N001" }
        return switch scaleEncoding {
        case .e8m0: "AGQ4E001"
        case .e4m4: "AGQ4M001"
        case .e3m4: "AGQ4S001"
        }
    }

    /// A versioned research format; deliberately not a native MLX checkpoint.
    public func artifactData() throws -> Data {
        guard rows <= UInt32.max, columns <= UInt32.max else {
            throw CompactInt4Error.invalid("Artifact dimensions exceed UInt32")
        }
        var data = Data(artifactFormat.utf8)
        Self.append(UInt32(rows), to: &data)
        Self.append(UInt32(columns), to: &data)
        data.append(contentsOf: codes)
        data.append(contentsOf: scaleBytes)
        for value in offsets { Self.append(value, to: &data) }
        for value in gains { Self.append(value, to: &data) }
        Self.append(Self.checksum(data), to: &data)
        return data
    }

    public static func readArtifact(_ data: Data) throws -> Self {
        let bytes = [UInt8](data)
        guard bytes.count >= 24,
            ["AGQ4E001", "AGQ4M001", "AGQ4S001", "AGQ4N001"].contains(String(decoding: bytes.prefix(8), as: UTF8.self))
        else {
            throw CompactInt4Error.invalid("Invalid compact INT4 artifact header")
        }
        func integer(_ start: Int, _ length: Int) -> UInt64 {
            (0..<length).reduce(0) { $0 | (UInt64(bytes[start + $1]) << ($1 * 8)) }
        }
        let rows = Int(integer(8, 4))
        let columns = Int(integer(12, 4))
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        // Counts must fit the actual payload before any model-sized allocation.
        guard !overflow, rows > 0, columns > 0, columns % 64 == 0,
            count / 2 <= bytes.count - 24, rows <= bytes.count / 2,
            count / 2 + count / 32 + count / 32 + rows * 2 == bytes.count - 24,
            checksum(data.prefix(data.count - 8)) == integer(bytes.count - 8, 8)
        else { throw CompactInt4Error.invalid("Artifact length, geometry or checksum mismatch") }
        var cursor = 16
        let codes = Array(bytes[cursor..<(cursor + count / 2)])
        cursor += count / 2
        let exponents = Array(bytes[cursor..<(cursor + count / 32)])
        cursor += count / 32
        let offsets = (0..<(count / 64)).map { UInt16(integer(cursor + $0 * 2, 2)) }
        cursor += count / 32
        let gains = (0..<rows).map { UInt16(integer(cursor + $0 * 2, 2)) }
        let encoding: ScaleEncoding
        switch bytes[4] {
        case 77, 78: encoding = .e4m4
        case 83: encoding = .e3m4
        default: encoding = .e8m0
        }
        return try Self(
            rows: rows, columns: columns, codes: codes, scaleBytes: exponents, offsets: offsets, gains: gains,
            scaleEncoding: encoding, codeEncoding: bytes[4] == 78 ? .normal16 : .signedInt4)
    }

    /// Publishes a complete file using an exclusive hard link; existing artifacts are never overwritten.
    public func writeArtifact(to destination: URL) throws {
        let parent = destination.deletingLastPathComponent()
        let stage = parent.appendingPathComponent(".compact-int4-\(UUID().uuidString).tmp")
        defer { try? FileManager.default.removeItem(at: stage) }
        try artifactData().write(to: stage, options: .withoutOverwriting)
        let status = stage.path.withCString { source in
            destination.path.withCString { target in link(source, target) }
        }
        guard status == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
    }

    private static func append<T: FixedWidthInteger>(_ integer: T, to data: inout Data) {
        for byte in 0..<MemoryLayout<T>.size {
            data.append(UInt8(truncatingIfNeeded: integer >> (byte * 8)))
        }
    }

    // FNV-1a detects accidental corruption; this is not an authenticity signature.
    private static func checksum(_ data: Data) -> UInt64 {
        data.reduce(14_695_981_039_346_656_037) { ($0 ^ UInt64($1)) &* 1_099_511_628_211 }
    }
}

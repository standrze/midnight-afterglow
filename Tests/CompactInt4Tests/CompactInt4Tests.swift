import Foundation
import XCTest

@testable import CompactInt4

final class CompactInt4Tests: XCTestCase {
    func testSignedNibblesIndependentGroupScalesAndOffsetCorrection() throws {
        let codes = (0..<64).map { ($0 % 16) - 8 }
        let bytes = stride(from: 0, to: 64, by: 2).map {
            UInt8(codes[$0] & 15) | (UInt8(codes[$0 + 1] & 15) << 4)
        }
        let matrix = try CompactInt4Matrix(
            rows: 1, columns: 64, codes: bytes, scaleBytes: [126, 129],
            offsets: [0x3f00], gains: [0x4000])
        let expected = (0..<64).map { k in Float(codes[k]) * (k < 32 ? 1 : 8) + 1 }
        XCTAssertEqual(matrix.reconstructed(), expected)
        let input = (0..<192).map { Float(($0 % 11) - 5) / 8 }
        let actual = try matrix.multiply(input, batch: 3)
        for sample in 0..<3 {
            let oracle = (0..<64).reduce(Float(0)) { $0 + input[sample * 64 + $1] * expected[$1] }
            XCTAssertEqual(actual[sample], oracle, accuracy: 1e-5)
        }
        XCTAssertEqual(matrix.payloadBytes, 38)
        XCTAssertEqual(matrix.bitsPerWeight, 4.75)
    }

    func testFitterAndArtifactRoundTripOnNonuniformRows() throws {
        let weights = fixture()
        let matrix = try CompactInt4Matrix.fit(weights, rows: 3, columns: 128)
        let decoded = try CompactInt4Matrix.readArtifact(matrix.artifactData())
        XCTAssertEqual(decoded.reconstructed(), matrix.reconstructed())
        XCTAssertEqual(decoded.codes, matrix.codes)
        let error = zip(weights, decoded.reconstructed()).reduce(0.0) { $0 + pow(Double($1.0 - $1.1), 2) }
        let energy = weights.reduce(0.0) { $0 + pow(Double($1), 2) }
        XCTAssertLessThan(error / energy, 0.02)
        let input = (0..<256).map { sin(Float($0) * 0.317) }
        let actual = try decoded.multiply(input, batch: 2)
        let dense = decoded.reconstructed()
        for sample in 0..<2 {
            for row in 0..<3 {
                let oracle = (0..<128).reduce(0.0) {
                    $0 + Double(input[sample * 128 + $1]) * Double(dense[row * 128 + $1])
                }
                XCTAssertEqual(Double(actual[sample * 3 + row]), oracle, accuracy: 2e-5)
            }
        }
    }

    func testRejectsCorruptionAndNeverOverwritesArtifacts() throws {
        let matrix = try CompactInt4Matrix.fit(fixture(), rows: 3, columns: 128)
        var data = try matrix.artifactData()
        data[19] ^= 1
        XCTAssertThrowsError(try CompactInt4Matrix.readArtifact(data))
        XCTAssertThrowsError(try CompactInt4Matrix.readArtifact(data.prefix(16)))
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }
        let destination = directory.appendingPathComponent("weights.agq4")
        try matrix.writeArtifact(to: destination)
        XCTAssertThrowsError(try matrix.writeArtifact(to: destination))
        XCTAssertEqual(try Data(contentsOf: destination), try matrix.artifactData())
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), ["weights.agq4"])
    }

    func testZeroConstantAndInvalidInputs() throws {
        for value: Float in [0, 0.25, -0.5] {
            let matrix = try CompactInt4Matrix.fit(.init(repeating: value, count: 64), rows: 1, columns: 64)
            for restored in matrix.reconstructed() { XCTAssertEqual(restored, value, accuracy: 1e-5) }
        }
        XCTAssertThrowsError(try CompactInt4Matrix.fit([.nan], rows: 1, columns: 64))
        XCTAssertThrowsError(try CompactInt4Matrix.fit([], rows: Int.max, columns: 64))
        XCTAssertThrowsError(
            try CompactInt4Matrix(
                rows: 1, columns: 64, codes: .init(repeating: 0, count: 32),
                scaleBytes: [255, 127], offsets: [0], gains: [0x3f80]))
        let matrix = try CompactInt4Matrix.fit(fixture(), rows: 3, columns: 128)
        XCTAssertThrowsError(try matrix.multiply([], batch: Int.max))
    }

    func testUnsignedScaleByteContractAndArtifactRoundTrip() throws {
        XCTAssertEqual(CompactInt4Matrix.scale(0, encoding: .e4m4), 1.0 / 128)
        XCTAssertEqual(CompactInt4Matrix.scale(0x70, encoding: .e4m4), 1)
        XCTAssertEqual(CompactInt4Matrix.scale(0x78, encoding: .e4m4), 1.5)
        XCTAssertEqual(CompactInt4Matrix.scale(255, encoding: .e4m4), 496)
        let matrix = try CompactInt4Matrix(
            rows: 1, columns: 64, codes: .init(repeating: 0x11, count: 32),
            scaleBytes: [0x70, 255], offsets: [0], gains: [0x3f80], scaleEncoding: .e4m4)
        let restored = try CompactInt4Matrix.readArtifact(matrix.artifactData())
        XCTAssertEqual(restored.scaleEncoding, .e4m4)
        XCTAssertEqual(restored.reconstructed(), [Float](repeating: 1, count: 32) + [Float](repeating: 496, count: 32))
        XCTAssertEqual(matrix.bitsPerWeight, 4.75)
    }

    func testSignedScaleByteContractAndArtifactRoundTrip() throws {
        for byte in UInt16(0)...255 {
            let sign: Float = byte & 128 == 0 ? 1 : -1
            let exponent = Int((byte >> 4) & 7) - 3
            let expected = sign * (1 + Float(byte & 15) / 16) * pow(2, Float(exponent))
            XCTAssertEqual(CompactInt4Matrix.scale(UInt8(byte), encoding: .e3m4), expected)
        }
        let matrix = try CompactInt4Matrix(
            rows: 1, columns: 64, codes: .init(repeating: 0x11, count: 32),
            scaleBytes: [0x30, 255], offsets: [0], gains: [0x3f80], scaleEncoding: .e3m4)
        let restored = try CompactInt4Matrix.readArtifact(matrix.artifactData())
        XCTAssertEqual(restored.scaleEncoding, .e3m4)
        XCTAssertEqual(restored.artifactFormat, "AGQ4S001")
        XCTAssertEqual(restored.reconstructed(), [Float](repeating: 1, count: 32) + [Float](repeating: -31, count: 32))
        XCTAssertThrowsError(try CompactInt4Matrix.fit(fixture(), rows: 3, columns: 128, scaleEncoding: .e3m4))
    }

    func testNormal16StoredCodebookAndArtifactContract() throws {
        let expectedBook: [Float] = [
            -1.8623046875, -1.318359375, -1.009765625, -0.7763671875, -0.5791015625, -0.40234375, -0.2371826171875,
            -0.07843017578125, 0.07843017578125, 0.2371826171875, 0.40234375, 0.5791015625, 0.7763671875, 1.009765625,
            1.318359375, 1.8623046875,
        ]
        let bytes = stride(from: 0, to: 64, by: 2).map { UInt8($0 % 16) | UInt8(($0 + 1) % 16) << 4 }
        let matrix = try CompactInt4Matrix(
            rows: 1, columns: 64, codes: bytes, scaleBytes: [0x70, 0x78], offsets: [0x3e80], gains: [0x4000],
            scaleEncoding: .e4m4, codeEncoding: .normal16)
        let expected = (0..<64).map { (expectedBook[$0 % 16] * ($0 < 32 ? 1 : 1.5) + 0.25) * 2 }
        XCTAssertEqual(matrix.reconstructed(), expected)
        let loaded = try CompactInt4Matrix.readArtifact(matrix.artifactData())
        XCTAssertEqual(loaded.codeEncoding, .normal16)
        XCTAssertEqual(loaded.scaleEncoding, .e4m4)
        XCTAssertEqual(loaded.artifactFormat, "AGQ4N001")
        XCTAssertEqual(loaded.reconstructed(), expected)
        XCTAssertEqual(try loaded.artifactData(), try matrix.artifactData())
        XCTAssertEqual(matrix.payloadBytes, 38)
        let input = (0..<192).map { sin(Float($0) * 0.19) }
        let actual = try loaded.multiply(input, batch: 3)
        for sample in 0..<3 {
            let oracle = (0..<64).reduce(0.0) { $0 + Double(input[sample * 64 + $1]) * Double(expected[$1]) }
            XCTAssertEqual(Double(actual[sample]), oracle, accuracy: 2e-5)
        }
        XCTAssertThrowsError(
            try CompactInt4Matrix(
                rows: 1, columns: 64, codes: bytes, scaleBytes: [127, 127], offsets: [0], gains: [0x3f80],
                scaleEncoding: .e8m0, codeEncoding: .normal16))
        #if canImport(Metal)
            XCTAssertThrowsError(
                try CompactInt4Metal(backend: .nativeInt4, scaleEncoding: .e4m4, codeEncoding: .normal16))
        #endif
    }

    private func fixture() -> [Float] {
        (0..<384).map { k in
            let amplitude = Float(k / 128 + 1) * 0.1
            let wave = sin(Float(k) * 0.123) * amplitude
            return wave + Float(k % 7) * 0.005
        }
    }
}

import Foundation
import QuantizerSupport
import XCTest

final class QuantizerOutputTests: XCTestCase {
    func testShardSizeRejectsOverflowAndSubByteValues() throws {
        XCTAssertEqual(try QuantizerShardSize.bytes(fromGiB: 5), 5_368_709_120)
        XCTAssertEqual(try QuantizerShardSize.bytes(fromGiB: 1.0 / 1_073_741_824), 1)
        for value in [Double.nan, .infinity, -.infinity, 0, -1, 1e-20, 8_589_934_592, 1e300] {
            XCTAssertThrowsError(try QuantizerShardSize.bytes(fromGiB: value), "value=\(value)")
        }
        XCTAssertGreaterThan(try QuantizerShardSize.bytes(fromGiB: 8_589_934_591), 0)
    }

    func testPreflightRejectsExistingOutputAndSourceAliases() throws {
        try withWorkspace { source, output in
            let manager = FileManager.default
            try manager.createDirectory(at: output, withIntermediateDirectories: false)
            XCTAssertThrowsError(
                try QuantizerOutputTransaction.validate(
                    sourceDirectory: source, destinationDirectory: output, overwrite: false))
            XCTAssertNoThrow(
                try QuantizerOutputTransaction.validate(
                    sourceDirectory: source, destinationDirectory: output, overwrite: true))
            for overlapping in [source, source.appendingPathComponent("output"), source.deletingLastPathComponent()] {
                XCTAssertThrowsError(
                    try QuantizerOutputTransaction.validate(
                        sourceDirectory: source, destinationDirectory: overlapping, overwrite: true))
            }
            let alias = source.deletingLastPathComponent().appendingPathComponent("source-alias")
            try manager.createSymbolicLink(at: alias, withDestinationURL: source)
            XCTAssertThrowsError(
                try QuantizerOutputTransaction.validate(
                    sourceDirectory: source, destinationDirectory: alias, overwrite: true))
            XCTAssertThrowsError(
                try QuantizerOutputTransaction.validate(
                    sourceDirectory: source, destinationDirectory: alias.appendingPathComponent("child"),
                    overwrite: true))
        }
    }

    func testFailedConversionPreservesPreviousOutputAndCleansPartial() throws {
        try withWorkspace { source, output in
            try writeCheckpoint("previous", at: output)
            let transaction = try QuantizerOutputTransaction(
                sourceDirectory: source, destinationDirectory: output, overwrite: true)
            try writeCheckpoint("incomplete", at: transaction.stagingDirectory)
            XCTAssertEqual(try readCheckpoint(at: output), "previous")
            transaction.cleanup()
            XCTAssertEqual(try readCheckpoint(at: output), "previous")
            XCTAssertFalse(FileManager.default.fileExists(atPath: transaction.stagingDirectory.path))
        }
    }

    func testCompletedConversionReplacesOutputOnlyAtCommit() throws {
        try withWorkspace { source, output in
            try writeCheckpoint("previous", at: output)
            let transaction = try QuantizerOutputTransaction(
                sourceDirectory: source, destinationDirectory: output, overwrite: true)
            defer { transaction.cleanup() }
            try writeCheckpoint("complete", at: transaction.stagingDirectory)
            XCTAssertEqual(try readCheckpoint(at: output), "previous")
            XCTAssertNil(try transaction.commit())
            XCTAssertEqual(try readCheckpoint(at: output), "complete")
            XCTAssertFalse(FileManager.default.fileExists(atPath: transaction.stagingDirectory.path))
            let names = try FileManager.default.contentsOfDirectory(atPath: output.deletingLastPathComponent().path)
            XCTAssertFalse(names.contains { $0.contains(".backup-") })
        }
    }

    func testPublicationFailureRestoresPreviousOutput() throws {
        try withWorkspace { source, output in
            try writeCheckpoint("previous", at: output)
            let transaction = try QuantizerOutputTransaction(
                sourceDirectory: source, destinationDirectory: output, overwrite: true)
            // A missing staging directory simulates publication failing after the
            // previous output has been moved aside.
            XCTAssertThrowsError(try transaction.commit())
            XCTAssertEqual(try readCheckpoint(at: output), "previous")
        }
    }

    func testConcurrentDestinationIsPreservedWithoutOverwrite() throws {
        try withWorkspace { source, output in
            let transaction = try QuantizerOutputTransaction(
                sourceDirectory: source, destinationDirectory: output, overwrite: false)
            defer { transaction.cleanup() }
            try writeCheckpoint("complete", at: transaction.stagingDirectory)
            try writeCheckpoint("another conversion", at: output)
            XCTAssertThrowsError(try transaction.commit())
            XCTAssertEqual(try readCheckpoint(at: output), "another conversion")
        }
    }

    func testNewOutputAppearsOnlyAfterCommit() throws {
        try withWorkspace { source, output in
            let transaction = try QuantizerOutputTransaction(
                sourceDirectory: source, destinationDirectory: output, overwrite: false)
            defer { transaction.cleanup() }
            try writeCheckpoint("complete", at: transaction.stagingDirectory)
            XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
            XCTAssertNil(try transaction.commit())
            XCTAssertEqual(try readCheckpoint(at: output), "complete")
        }
    }

    private func withWorkspace(_ body: (URL, URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("wick-output-test-\(UUID())")
        let source = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(source, root.appendingPathComponent("output"))
    }

    private func writeCheckpoint(_ value: String, at directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try Data(value.utf8).write(to: directory.appendingPathComponent("model.safetensors"))
    }

    private func readCheckpoint(at directory: URL) throws -> String {
        String(decoding: try Data(contentsOf: directory.appendingPathComponent("model.safetensors")), as: UTF8.self)
    }
}

import Foundation
import Testing

@testable import MistralActivationScaleSearchCore

struct IndexedSafetensorsFingerprintTests {
  @Test("Fingerprint covers complete files, ignores JSON key order, and includes indexed names")
  func fullFilesAndNames() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("weight-fingerprint-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let payload = root.appendingPathComponent("first.safetensors")
    // More than one read chunk ensures the tail is covered rather than sampled.
    var bytes = Data(repeating: 17, count: IndexedSafetensorsFingerprint.chunkBytes + 32)
    try bytes.write(to: payload)
    let indexA = Data(#"{"weight_map":{"one":"first.safetensors","two":"first.safetensors"}}"#.utf8)
    let indexB = Data(#"{"weight_map":{"two":"first.safetensors","one":"first.safetensors"}}"#.utf8)
    let first = try IndexedSafetensorsFingerprint.compute(directory: root, indexData: indexA)
    #expect(try IndexedSafetensorsFingerprint.compute(directory: root, indexData: indexB) == first)
    bytes[bytes.count - 1] ^= 1
    try bytes.write(to: payload)
    #expect(try IndexedSafetensorsFingerprint.compute(directory: root, indexData: indexA) != first)
    bytes[bytes.count - 1] ^= 1
    try bytes.write(to: payload)
    try FileManager.default.copyItem(at: payload, to: root.appendingPathComponent("renamed.safetensors"))
    let renamed = Data(#"{"weight_map":{"one":"renamed.safetensors","two":"renamed.safetensors"}}"#.utf8)
    #expect(try IndexedSafetensorsFingerprint.compute(directory: root, indexData: renamed) != first)
  }
}

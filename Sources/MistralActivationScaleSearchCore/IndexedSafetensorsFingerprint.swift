import Foundation

/// Reproducibility identity for the complete indexed source files. FNV-1a is
/// deliberately noncryptographic: this detects changed inputs, not adversarial
/// hash collisions or a publisher's authenticity.
public enum IndexedSafetensorsFingerprint {
  public static let method = "indexed-safetensors-full-content-fnv1a64-v1"
  public static let chunkBytes = 8 * 1_024 * 1_024

  public static func compute(directory: URL, indexData: Data? = nil) throws -> String {
    let root = directory.standardizedFileURL
    let data = try indexData ?? Data(contentsOf: root.appendingPathComponent("model.safetensors.index.json"))
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let map = json["weight_map"] as? [String: String], !map.isEmpty
    else { throw FingerprintError.invalidSource("missing nonempty weight_map") }
    var hasher = FNV1a64()
    hasher.update(Array((method + "\0").utf8))
    let shards = Set(map.values).sorted()
    hasher.updateInteger(UInt64(shards.count))
    for name in shards {
      let url = root.appendingPathComponent(name).standardizedFileURL
      // Validate the logical index path; ordinary HF snapshots may then resolve
      // the in-directory filename to a shared blob outside the snapshot.
      guard !name.isEmpty, !name.hasPrefix("/"), url.path.hasPrefix(root.path + "/") else {
        throw FingerprintError.invalidSource("indexed shard escapes source directory")
      }
      let file = try FileHandle(forReadingFrom: url)
      defer { try? file.close() }
      let size = try file.seekToEnd()
      try file.seek(toOffset: 0)
      let nameBytes = Array(name.utf8)
      hasher.updateInteger(UInt64(nameBytes.count))
      hasher.update(nameBytes)
      hasher.updateInteger(size)
      var remaining = size
      while remaining > 0 {
        let request = Int(min(UInt64(chunkBytes), remaining))
        let consumed: Int
        #if canImport(ObjectiveC)
          consumed = try autoreleasepool {
            try consumeChunk(file, request: request, hasher: &hasher)
          }
        #else
          consumed = try consumeChunk(file, request: request, hasher: &hasher)
        #endif
        guard consumed > 0 else { throw FingerprintError.invalidSource("source shard shrank while fingerprinting") }
        remaining -= UInt64(consumed)
      }
      guard try file.seekToEnd() == size else {
        throw FingerprintError.invalidSource("source shard changed size while fingerprinting")
      }
    }
    return String(format: "fnv1a64:%016llx", hasher.value)
  }

  private static func consumeChunk(_ file: FileHandle, request: Int, hasher: inout FNV1a64) throws -> Int {
    guard let data = try file.read(upToCount: request) else { return 0 }
    data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
      for byte in bytes { hasher.value = (hasher.value ^ UInt64(byte)) &* 1_099_511_628_211 }
    }
    return data.count
  }

  private struct FNV1a64 {
    var value: UInt64 = 14_695_981_039_346_656_037
    mutating func update(_ bytes: [UInt8]) {
      for byte in bytes { value = (value ^ UInt64(byte)) &* 1_099_511_628_211 }
    }
    mutating func updateInteger(_ value: UInt64) {
      for shift in stride(from: 0, through: 56, by: 8) {
        self.value = (self.value ^ ((value >> shift) & 255)) &* 1_099_511_628_211
      }
    }
  }

  public enum FingerprintError: Error, LocalizedError {
    case invalidSource(String)
    public var errorDescription: String? {
      switch self {
      case .invalidSource(let message): "Cannot fingerprint source weights: \(message)"
      }
    }
  }
}

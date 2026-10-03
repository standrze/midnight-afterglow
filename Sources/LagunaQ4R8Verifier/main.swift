import ArgumentParser
import Foundation

private struct SafetensorsIndex: Decodable {
  var weightMap: [String: String]

  enum CodingKeys: String, CodingKey {
    case weightMap = "weight_map"
  }
}

private struct ScaleSearchProvenance: Decodable {
  var algorithm: String
  var bits: Int
  var groupSize: Int
  var q4ModulesRescored: Int
  var q8ModulesPreserved: Int
  var standardQ4EmbeddingsPreserved: Int

  enum CodingKeys: String, CodingKey {
    case algorithm, bits
    case groupSize = "group_size"
    case q4ModulesRescored = "q4_modules_rescored"
    case q8ModulesPreserved = "q8_modules_preserved"
    case standardQ4EmbeddingsPreserved = "standard_q4_embeddings_preserved"
  }
}

private struct TensorDescriptor {
  var dtype: String
  var shape: [Int]
  var start: UInt64
  var end: UInt64

  var byteCount: UInt64 { end - start }

  func sameLayout(as other: TensorDescriptor) -> Bool {
    dtype == other.dtype && shape == other.shape && byteCount == other.byteCount
  }
}

private struct SafetensorsShard {
  var url: URL
  var dataStart: UInt64
  var tensors: [String: TensorDescriptor]
}

private struct VerificationReport: Encodable {
  var format = 1
  var status = "verified"
  var verifiedAt: String
  var standardTemplate: String
  var searchedCandidate: String
  var algorithm: String
  var indexedTensors: Int
  var shards: Int
  var tensorPayloadBytes: UInt64
  var q4Modules: Int
  var q4ModulesChanged: Int
  var q4Tensors: Int
  var q4TensorsChanged: Int
  var preservedQ8Modules: Int
  var preservedQ4Embeddings: Int
  var preservedTensors: Int
  var preservedTensorBytes: UInt64
  var identicalSidecars: Int

  enum CodingKeys: String, CodingKey {
    case format, status, algorithm, shards
    case verifiedAt = "verified_at"
    case standardTemplate = "standard_template"
    case searchedCandidate = "searched_candidate"
    case indexedTensors = "indexed_tensors"
    case tensorPayloadBytes = "tensor_payload_bytes"
    case q4Modules = "q4_modules"
    case q4ModulesChanged = "q4_modules_changed"
    case q4Tensors = "q4_tensors"
    case q4TensorsChanged = "q4_tensors_changed"
    case preservedQ8Modules = "preserved_q8_modules"
    case preservedQ4Embeddings = "preserved_q4_embeddings"
    case preservedTensors = "preserved_tensors"
    case preservedTensorBytes = "preserved_tensor_bytes"
    case identicalSidecars = "identical_sidecars"
  }
}

private enum VerificationError: Error, LocalizedError {
  case invalidInput(String)
  case malformedSafetensors(String)
  case mismatch(String)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let message): "Invalid verification input: \(message)"
    case .malformedSafetensors(let message): "Malformed safetensors file: \(message)"
    case .mismatch(let message): "Checkpoint verification failed: \(message)"
    }
  }
}

@main
private struct LagunaQ4R8Verifier: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: CommandLine.arguments.first?.split(separator: "/").last.map(String.init)
      ?? "wick-laguna-q4r8-verify",
    abstract:
      "Prove that a searched Laguna Q4R8 checkpoint preserves its layout while changing only Q4 modules."
  )

  @Argument(help: "Standard Laguna MLX Q4R8 checkpoint used as the immutable template.")
  var template: String

  @Argument(help: "ScaleSearch Laguna MLX Q4R8 checkpoint to verify.")
  var candidate: String

  @Option(
    name: .customLong("report"),
    help: "Write a JSON verification report after every check succeeds; the path must not exist."
  )
  var report: String?

  mutating func run() throws {
    let templateURL = URL(fileURLWithPath: template).standardizedFileURL
    let candidateURL = URL(fileURLWithPath: candidate).standardizedFileURL
    try validateDirectory(templateURL, label: "template")
    try validateDirectory(candidateURL, label: "candidate")
    guard templateURL != candidateURL else {
      throw VerificationError.invalidInput("template and candidate must be distinct")
    }

    let templateIndexURL = templateURL.appendingPathComponent("model.safetensors.index.json")
    let candidateIndexURL = candidateURL.appendingPathComponent("model.safetensors.index.json")
    let templateIndexData = try Data(contentsOf: templateIndexURL)
    let candidateIndexData = try Data(contentsOf: candidateIndexURL)
    guard templateIndexData == candidateIndexData else {
      throw VerificationError.mismatch("safetensors index files are not byte-identical")
    }
    let index = try JSONDecoder().decode(SafetensorsIndex.self, from: candidateIndexData)
    guard !index.weightMap.isEmpty else {
      throw VerificationError.invalidInput("safetensors index is empty")
    }

    let quantizedModules = index.weightMap.keys.compactMap { key -> String? in
      guard key.hasSuffix(".weight") else { return nil }
      let module = String(key.dropLast(".weight".count))
      guard index.weightMap["\(module).scales"] != nil,
        index.weightMap["\(module).biases"] != nil
      else { return nil }
      return module
    }.sorted()
    let q8Modules = Set(quantizedModules.filter(Self.isQ8Router))
    let embeddingModules = Set(quantizedModules.filter(Self.isStandardEmbedding))
    let q4Modules = Set(quantizedModules).subtracting(q8Modules).subtracting(embeddingModules)
    guard !q4Modules.isEmpty else {
      throw VerificationError.invalidInput("no affine Q4 modules were found")
    }

    var q4TensorToModule = [String: String]()
    for module in q4Modules {
      for suffix in ["weight", "scales", "biases"] {
        q4TensorToModule["\(module).\(suffix)"] = module
      }
    }

    let provenanceURL = candidateURL.appendingPathComponent("q4r8-scale-search.json")
    let provenance = try JSONDecoder().decode(
      ScaleSearchProvenance.self,
      from: Data(contentsOf: provenanceURL)
    )
    guard provenance.algorithm == "q4r8_affine_scale_search_ls2",
      provenance.bits == 4,
      provenance.groupSize == 64,
      provenance.q4ModulesRescored == q4Modules.count,
      provenance.q8ModulesPreserved == q8Modules.count,
      provenance.standardQ4EmbeddingsPreserved == embeddingModules.count
    else {
      throw VerificationError.mismatch("ScaleSearch provenance does not match the indexed layout")
    }

    let shardNames = Set(index.weightMap.values).sorted()
    var changedQ4Modules = Set<String>()
    var q4Tensors = 0
    var changedQ4Tensors = 0
    var preservedTensors = 0
    var payloadBytes: UInt64 = 0
    var preservedBytes: UInt64 = 0

    for (offset, shardName) in shardNames.enumerated() {
      let expectedKeys = Set(index.weightMap.compactMap { key, shard in
        shard == shardName ? key : nil
      })
      let standard = try parseSafetensors(
        at: templateURL.appendingPathComponent(shardName))
      let searched = try parseSafetensors(
        at: candidateURL.appendingPathComponent(shardName))
      guard Set(standard.tensors.keys) == expectedKeys else {
        throw VerificationError.mismatch("template shard \(shardName) does not match its index")
      }
      guard Set(searched.tensors.keys) == expectedKeys else {
        throw VerificationError.mismatch("candidate shard \(shardName) does not match its index")
      }

      let standardHandle = try FileHandle(forReadingFrom: standard.url)
      let searchedHandle = try FileHandle(forReadingFrom: searched.url)
      defer {
        try? standardHandle.close()
        try? searchedHandle.close()
      }
      for key in expectedKeys.sorted() {
        guard let standardTensor = standard.tensors[key],
          let searchedTensor = searched.tensors[key]
        else {
          throw VerificationError.mismatch("missing tensor \(key) in \(shardName)")
        }
        guard standardTensor.sameLayout(as: searchedTensor) else {
          throw VerificationError.mismatch(
            "layout changed for \(key): standard dtype=\(standardTensor.dtype) shape=\(standardTensor.shape) bytes=\(standardTensor.byteCount), "
              + "searched dtype=\(searchedTensor.dtype) shape=\(searchedTensor.shape) bytes=\(searchedTensor.byteCount)"
          )
        }

        let bytes = standardTensor.byteCount
        payloadBytes += bytes
        let equal = try tensorBytesEqual(
          standardHandle: standardHandle,
          standardOffset: standard.dataStart + standardTensor.start,
          searchedHandle: searchedHandle,
          searchedOffset: searched.dataStart + searchedTensor.start,
          byteCount: bytes
        )
        if let module = q4TensorToModule[key] {
          q4Tensors += 1
          if !equal {
            changedQ4Tensors += 1
            changedQ4Modules.insert(module)
          }
        } else {
          guard equal else {
            throw VerificationError.mismatch("preserved tensor changed: \(key)")
          }
          preservedTensors += 1
          preservedBytes += bytes
        }
      }
      print("[\(offset + 1)/\(shardNames.count)] verified \(shardName) (\(expectedKeys.count) tensors)")
    }

    let unchangedQ4Modules = q4Modules.subtracting(changedQ4Modules)
    guard unchangedQ4Modules.isEmpty else {
      throw VerificationError.mismatch(
        "\(unchangedQ4Modules.count) searched Q4 module(s) are identical to the standard template: "
          + unchangedQ4Modules.sorted().prefix(5).joined(separator: ", ")
      )
    }
    guard q4Tensors == q4Modules.count * 3 else {
      throw VerificationError.mismatch(
        "expected \(q4Modules.count * 3) Q4 tensors but classified \(q4Tensors)"
      )
    }

    let identicalSidecars = try verifySidecars(templateURL: templateURL, candidateURL: candidateURL)
    print("Every searched Q4 module differs from the standard template (\(changedQ4Modules.count)/\(q4Modules.count)).")
    print("All preserved tensors are byte-for-byte identical (\(preservedTensors) tensors, \(preservedBytes) bytes).")
    print("All copied sidecars are byte-for-byte identical (\(identicalSidecars) files).")

    let verification = VerificationReport(
      verifiedAt: ISO8601DateFormatter().string(from: Date()),
      standardTemplate: templateURL.path,
      searchedCandidate: candidateURL.path,
      algorithm: provenance.algorithm,
      indexedTensors: index.weightMap.count,
      shards: shardNames.count,
      tensorPayloadBytes: payloadBytes,
      q4Modules: q4Modules.count,
      q4ModulesChanged: changedQ4Modules.count,
      q4Tensors: q4Tensors,
      q4TensorsChanged: changedQ4Tensors,
      preservedQ8Modules: q8Modules.count,
      preservedQ4Embeddings: embeddingModules.count,
      preservedTensors: preservedTensors,
      preservedTensorBytes: preservedBytes,
      identicalSidecars: identicalSidecars
    )
    if let report {
      let reportURL = URL(fileURLWithPath: report).standardizedFileURL
      guard !FileManager.default.fileExists(atPath: reportURL.path) else {
        throw VerificationError.invalidInput("report already exists: \(reportURL.path)")
      }
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(verification).write(to: reportURL, options: .atomic)
      print("Wrote verification report: \(reportURL.path)")
    }
  }

  private func parseSafetensors(at url: URL) throws -> SafetensorsShard {
    guard let fileSize = (try FileManager.default.attributesOfItem(atPath: url.path)[.size]
      as? NSNumber)?.uint64Value,
      fileSize >= 8
    else {
      throw VerificationError.malformedSafetensors("missing or undersized file: \(url.path)")
    }
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let prefix = try readExactly(handle, count: 8, context: url.lastPathComponent)
    var headerLength: UInt64 = 0
    for (offset, byte) in prefix.enumerated() {
      headerLength |= UInt64(byte) << UInt64(offset * 8)
    }
    guard headerLength > 0, headerLength <= fileSize - 8, headerLength <= UInt64(Int.max) else {
      throw VerificationError.malformedSafetensors(
        "invalid header length \(headerLength) in \(url.lastPathComponent)")
    }
    let headerData = try readExactly(
      handle, count: Int(headerLength), context: url.lastPathComponent)
    guard let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
      throw VerificationError.malformedSafetensors(
        "header is not a JSON object in \(url.lastPathComponent)")
    }

    let dataStart = 8 + headerLength
    let payloadBytes = fileSize - dataStart
    var tensors = [String: TensorDescriptor]()
    for (key, rawValue) in object where key != "__metadata__" {
      guard let value = rawValue as? [String: Any],
        let dtype = value["dtype"] as? String,
        let rawShape = value["shape"] as? [NSNumber],
        let offsets = value["data_offsets"] as? [NSNumber],
        offsets.count == 2
      else {
        throw VerificationError.malformedSafetensors(
          "invalid descriptor for \(key) in \(url.lastPathComponent)")
      }
      let shape = rawShape.map(\.intValue)
      guard shape.allSatisfy({ $0 >= 0 }) else {
        throw VerificationError.malformedSafetensors(
          "negative shape for \(key) in \(url.lastPathComponent)")
      }
      let descriptor = TensorDescriptor(
        dtype: dtype,
        shape: shape,
        start: offsets[0].uint64Value,
        end: offsets[1].uint64Value
      )
      guard descriptor.start <= descriptor.end, descriptor.end <= payloadBytes else {
        throw VerificationError.malformedSafetensors(
          "out-of-range payload for \(key) in \(url.lastPathComponent)")
      }
      guard tensors.updateValue(descriptor, forKey: key) == nil else {
        throw VerificationError.malformedSafetensors(
          "duplicate tensor \(key) in \(url.lastPathComponent)")
      }
    }
    guard !tensors.isEmpty else {
      throw VerificationError.malformedSafetensors("no tensors in \(url.lastPathComponent)")
    }

    let ordered = tensors.sorted { lhs, rhs in
      lhs.value.start == rhs.value.start ? lhs.key < rhs.key : lhs.value.start < rhs.value.start
    }
    var previousEnd: UInt64 = 0
    for (key, descriptor) in ordered {
      guard descriptor.start >= previousEnd else {
        throw VerificationError.malformedSafetensors(
          "overlapping payload for \(key) in \(url.lastPathComponent)")
      }
      previousEnd = descriptor.end
    }
    guard previousEnd == payloadBytes else {
      throw VerificationError.malformedSafetensors(
        "tensor payload ends at \(previousEnd), file payload is \(payloadBytes) in \(url.lastPathComponent)"
      )
    }
    return SafetensorsShard(url: url, dataStart: dataStart, tensors: tensors)
  }

  private func tensorBytesEqual(
    standardHandle: FileHandle,
    standardOffset: UInt64,
    searchedHandle: FileHandle,
    searchedOffset: UInt64,
    byteCount: UInt64
  ) throws -> Bool {
    try standardHandle.seek(toOffset: standardOffset)
    try searchedHandle.seek(toOffset: searchedOffset)
    let chunkSize: UInt64 = 8 * 1_024 * 1_024
    var remaining = byteCount
    while remaining > 0 {
      let count = Int(min(remaining, chunkSize))
      let standard = try readExactly(standardHandle, count: count, context: "template tensor")
      let searched = try readExactly(searchedHandle, count: count, context: "candidate tensor")
      if standard != searched { return false }
      remaining -= UInt64(count)
    }
    return true
  }

  private func readExactly(_ handle: FileHandle, count: Int, context: String) throws -> Data {
    guard let data = try handle.read(upToCount: count), data.count == count else {
      throw VerificationError.malformedSafetensors(
        "short read while reading \(context); expected \(count) bytes")
    }
    return data
  }

  private func verifySidecars(templateURL: URL, candidateURL: URL) throws -> Int {
    let ignored = Set(["q4r8-scale-search.json", "q4r8-verification.json"])
    let templateNames = try sidecarNames(at: templateURL, ignoring: ignored)
    let candidateNames = try sidecarNames(at: candidateURL, ignoring: ignored)
    guard templateNames == candidateNames else {
      let missing = templateNames.subtracting(candidateNames).sorted()
      let unexpected = candidateNames.subtracting(templateNames).sorted()
      throw VerificationError.mismatch(
        "sidecar sets differ; missing=\(missing) unexpected=\(unexpected)"
      )
    }
    for name in templateNames.sorted() {
      let standard = try Data(contentsOf: templateURL.appendingPathComponent(name))
      let searched = try Data(contentsOf: candidateURL.appendingPathComponent(name))
      guard standard == searched else {
        throw VerificationError.mismatch("copied sidecar changed: \(name)")
      }
    }
    return templateNames.count
  }

  private func sidecarNames(at directory: URL, ignoring: Set<String>) throws -> Set<String> {
    let items = try FileManager.default.contentsOfDirectory(
      at: directory,
      includingPropertiesForKeys: [.isRegularFileKey],
      options: [.skipsHiddenFiles]
    )
    return Set(try items.compactMap { item in
      let values = try item.resourceValues(forKeys: [.isRegularFileKey])
      guard values.isRegularFile == true,
        item.pathExtension != "safetensors",
        !ignoring.contains(item.lastPathComponent)
      else { return nil }
      return item.lastPathComponent
    })
  }

  private func validateDirectory(_ url: URL, label: String) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw VerificationError.invalidInput("\(label) directory does not exist: \(url.path)")
    }
  }

  private static func isQ8Router(_ module: String) -> Bool {
    module.hasSuffix(".mlp.gate.proj")
  }

  private static func isStandardEmbedding(_ module: String) -> Bool {
    module == "language_model.model.embed_tokens"
  }
}

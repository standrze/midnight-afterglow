import Foundation
import MLX
import MistralActivationScaleSearchCore

struct LagunaActivationInputError: Error, LocalizedError {
  var message: String
  var errorDescription: String? { "Invalid Laguna activation statistics: \(message)" }
}

struct LagunaActivationStatistics {
  static let momentSuffix = ".input_second_moment"
  static let countSuffix = ".expert_position_count"
  var moments: [String: MLXArray]
  var counts: [String: [Int]]
  var metadata: [String: String]
  var minimumExpertPositions: Int
  var url: URL

  static func fingerprint(_ data: Data) -> String {
    var hash: UInt64 = 14_695_981_039_346_656_037
    for byte in data { hash = (hash ^ UInt64(byte)) &* 1_099_511_628_211 }
    return String(format: "fnv1a64:%016llx", hash)
  }

  static func load(
    from url: URL, sourceURL: URL, sourceConfig: Data, sourceIndex: Data,
    sourceWeightFingerprint: String? = nil
  ) throws -> Self {
    let (arrays, metadata) = try loadArraysAndMetadata(url: url, stream: .cpu)
    guard metadata["format"] == "laguna_expert_activation_stats_v1",
      metadata["algorithm"] == "expert_conditional_input_channel_second_moment",
      metadata["model_type"] == "laguna", metadata["dtype"] == "float32",
      metadata["add_special_tokens"] == "true",
      metadata["source_config_fingerprint"] == fingerprint(sourceConfig),
      metadata["source_index_fingerprint"] == fingerprint(sourceIndex),
      let recordedSource = metadata["source_model"],
      URL(fileURLWithPath: recordedSource).standardizedFileURL.resolvingSymlinksInPath()
        == sourceURL.standardizedFileURL.resolvingSymlinksInPath(),
      let corpus = metadata["corpus_fingerprint"], !corpus.isEmpty,
      let tokens = metadata["token_id_fingerprint"], !tokens.isEmpty,
      let observed = metadata["observed_token_count"].flatMap(Int.init), observed > 0,
      let denseModules = metadata["module_count"].flatMap(Int.init), denseModules > 0,
      let expertModules = metadata["expert_module_count"].flatMap(Int.init), expertModules > 0,
      let minimum = metadata["minimum_expert_positions"].flatMap(Int.init), minimum > 0,
      metadata["insufficient_coverage_policy"] == "retain_template_expert",
      metadata["router_weighting"] == "conditional_on_selection_no_routing_score_weight"
    else {
      throw LagunaActivationInputError(message: "metadata does not match the BF16 source or supported conditional objective")
    }
    guard metadata["source_weight_fingerprint_method"] == IndexedSafetensorsFingerprint.method,
      let recordedWeightFingerprint = metadata["source_weight_fingerprint"], !recordedWeightFingerprint.isEmpty
    else { throw LagunaActivationInputError(message: "missing full source weight-content identity") }
    let actualWeightFingerprint = try sourceWeightFingerprint
      ?? IndexedSafetensorsFingerprint.compute(directory: sourceURL, indexData: sourceIndex)
    guard recordedWeightFingerprint == actualWeightFingerprint else {
      throw LagunaActivationInputError(message: "source weight payload changed since activation collection")
    }
    for key in ["sample_token_fingerprints", "segment_token_fingerprints"] {
      guard let encoded = metadata[key],
        let fingerprints = try? JSONDecoder().decode([String].self, from: Data(encoded.utf8)),
        !fingerprints.isEmpty, fingerprints.allSatisfy({ !$0.isEmpty })
      else { throw LagunaActivationInputError(message: "missing exact sample/segment overlap provenance") }
    }
    var moments = [String: MLXArray]()
    var counts = [String: [Int]]()
    for (key, value) in arrays {
      if key.hasSuffix(momentSuffix) {
        let path = String(key.dropLast(momentSuffix.count))
        guard !path.isEmpty, value.dtype == .float32, [1, 2].contains(value.ndim),
          value.shape.allSatisfy({ $0 > 0 }), moments[path] == nil
        else { throw LagunaActivationInputError(message: "invalid moment tensor \(key)") }
        try MLX.checkedEval(value)
        guard value.asArray(Float.self).allSatisfy({ $0.isFinite && $0 >= 0 }) else {
          throw LagunaActivationInputError(message: "non-finite or negative moment tensor \(key)")
        }
        moments[path] = value
      } else if key.hasSuffix(countSuffix) {
        let path = String(key.dropLast(countSuffix.count))
        guard !path.isEmpty, value.dtype == .int32, value.ndim == 1, counts[path] == nil else {
          throw LagunaActivationInputError(message: "invalid expert count tensor \(key)")
        }
        try MLX.checkedEval(value)
        let values = value.asArray(Int32.self).map(Int.init)
        guard !values.isEmpty, values.allSatisfy({ $0 >= 0 && $0 <= observed }) else {
          throw LagunaActivationInputError(message: "expert counts exceed observed token coverage for \(path)")
        }
        counts[path] = values
      } else { throw LagunaActivationInputError(message: "unknown tensor \(key)") }
    }
    let dense = moments.filter { $0.value.ndim == 1 }
    let experts = moments.filter { $0.value.ndim == 2 }
    guard dense.count == denseModules, experts.count == expertModules,
      Set(experts.keys) == Set(counts.keys)
    else { throw LagunaActivationInputError(message: "module coverage metadata and tensors disagree") }
    for (path, moment) in experts {
      guard counts[path]?.count == moment.dim(0) else {
        throw LagunaActivationInputError(message: "expert count shape does not match \(path)")
      }
    }
    return Self(moments: moments, counts: counts, metadata: metadata,
      minimumExpertPositions: minimum, url: url)
  }

  /// Public templates split dense/shared and routed gate/up; the native
  /// collector fuses them. Both projections use the same recorded input, with
  /// expert-conditional matrices retained for routed projections.
  func momentPath(for module: String) throws -> String {
    if moments[module] != nil { return module }
    for suffix in [".gate_proj", ".up_proj"] where module.hasSuffix(suffix) {
      let fused = String(module.dropLast(suffix.count)) + ".gate_up_proj"
      let expectedRank = module.contains(".switch_mlp.") ? 2 : 1
      if moments[fused]?.ndim == expectedRank { return fused }
    }
    throw LagunaActivationInputError(message: "missing projection \(module)")
  }

  func requireDisjoint(from other: Self) throws {
    guard metadata["corpus_fingerprint"] != other.metadata["corpus_fingerprint"],
      metadata["token_id_fingerprint"] != other.metadata["token_id_fingerprint"]
    else { throw LagunaActivationInputError(message: "calibration and dev statistics must have distinct fingerprints") }
    for key in ["sample_token_fingerprints", "segment_token_fingerprints"] {
      guard let lhs = metadata[key], let rhs = other.metadata[key],
        let left = try? JSONDecoder().decode([String].self, from: Data(lhs.utf8)),
        let right = try? JSONDecoder().decode([String].self, from: Data(rhs.utf8)),
        !left.isEmpty, !right.isEmpty, Set(left).isDisjoint(with: Set(right))
      else { throw LagunaActivationInputError(message: "calibration and dev have overlapping exact token samples/segments") }
    }
  }
}

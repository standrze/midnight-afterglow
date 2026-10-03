import Foundation
import MLX
import MistralActivationScaleSearchCore
import MLXLMCommon
import Testing

@testable import LagunaScaleSearchCore

@Suite("Laguna activation-weighted checkpoint conversion", .serialized)
struct LagunaActivationRescoreTests {
  @Test("Fused and public split conversion preserve routers, embeddings, schema, and under-covered experts", arguments: [false, true])
  func conversionPreservesPolicyAndFallback(splitTemplate: Bool) throws {
    let fixture = try Fixture(splitTemplate: splitTemplate)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try LagunaScaleSearchRescorer.rescore(
      source: fixture.source.path, template: fixture.template.path, destination: fixture.output.path,
      expertBatch: 1, activationStats: fixture.calibration.path, validationStats: fixture.validation.path,
      cpu: true)
    let result = try MLX.loadArrays(url: fixture.output.appendingPathComponent("model.safetensors"), stream: .cpu)
    #expect(Set(result.keys) == Set(fixture.templateArrays.keys))
    for (key, expected) in fixture.templateArrays {
      let actual = try #require(result[key])
      #expect(actual.shape == expected.shape)
      #expect(actual.dtype == expected.dtype)
      #expect(actual.nbytes == expected.nbytes)
      if key.contains(".gate.proj.") || key.contains(".embed_tokens.") {
        #expect(MLX.arrayEqual(actual, expected).item(Bool.self))
      }
      if key.contains(".switch_mlp.") {
        #expect(MLX.arrayEqual(actual[1], expected[1]).item(Bool.self))
      }
    }
    #expect(try Data(contentsOf: fixture.output.appendingPathComponent("model.safetensors.index.json"))
      == Data(contentsOf: fixture.template.appendingPathComponent("model.safetensors.index.json")))
    let provenance = try #require(JSONSerialization.jsonObject(
      with: Data(contentsOf: fixture.output.appendingPathComponent("q4r8-scale-search.json"))) as? [String: Any])
    #expect(provenance["algorithm"] as? String == "laguna_q4r8_activation_weighted_scale_search")
    let sourceIndex = try #require(JSONSerialization.jsonObject(with: fixture.sourceIndexData) as? [String: Any])
    let sourceMap = try #require(sourceIndex["weight_map"] as? [String: String])
    #expect(provenance["source_tensors_released"] as? Int == sourceMap.count)
    #expect(provenance["source_tensors_remaining"] as? Int == 0)
    let peak = try #require(provenance["peak_mlx_memory_bytes"] as? Int)
    let preflightPeak = try #require(provenance["preflight_peak_mlx_memory_bytes"] as? Int)
    #expect(peak >= preflightPeak && preflightPeak >= 0)
    let retained = try #require(provenance["retained_template_experts"] as? [String: [Int]])
    for projection in splitTemplate ? ["gate_proj", "up_proj"] : ["gate_up_proj"] {
      #expect(retained[Fixture.switchPrefix + projection] == [1])
    }
    #expect(retained[Fixture.switchPrefix + "down_proj"] == [1])
  }

  @Test("Public split layout default LS2 conversion matches direct source quantization")
  func splitLayoutLS2() throws {
    let fixture = try Fixture(splitTemplate: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try LagunaScaleSearchRescorer.rescore(source: fixture.source.path, template: fixture.template.path,
      destination: fixture.output.path, expertBatch: 1, cpu: true)
    let result = try MLX.loadArrays(url: fixture.output.appendingPathComponent("model.safetensors"), stream: .cpu)
    let source = try MLX.loadArrays(url: fixture.source.appendingPathComponent("model.safetensors"), stream: .cpu)
    #expect(Set(result.keys) == Set(fixture.templateArrays.keys))
    for projection in ["gate_proj", "up_proj", "down_proj"] {
      for expert in 0..<2 {
        let original = try #require(source["model.layers.1.mlp.experts.\(expert).\(projection).weight"])
        let searched = try Device.withDefaultDevice(.cpu) {
          let arrays = q4AffineScaleSearchQuantized(original)
          try MLX.checkedEval(arrays.weight, arrays.scales, arrays.biases)
          return arrays
        }
        let module = Fixture.switchPrefix + projection
        for (suffix, expected) in [("weight", searched.weight), ("scales", searched.scales), ("biases", searched.biases)] {
          let actual = try #require(result[module + "." + suffix])
          #expect(MLX.arrayEqual(actual[expert], expected).item(Bool.self))
        }
      }
    }
    for (key, expected) in fixture.templateArrays where key.contains(".gate.proj.") || key.contains(".embed_tokens.") {
      #expect(MLX.arrayEqual(try #require(result[key]), expected).item(Bool.self))
    }
  }

  @Test("Group-128 standard and LS2 conversion halve Q4 metadata while preserving G64 routers and embedding", arguments: [true, false])
  func group128Conversion(standard: Bool) throws {
    let fixture = try Fixture(splitTemplate: true, inputWidth: 128)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try Device.withDefaultDevice(.cpu) {
      try LagunaScaleSearchRescorer.rescore(source: fixture.source.path, template: fixture.template.path,
        destination: fixture.output.path, expertBatch: 1, groupSize: 128, standardQ4: standard, cpu: true)
      let result = try MLX.loadArrays(url: fixture.output.appendingPathComponent("model.safetensors"), stream: .cpu)
      #expect(Set(result.keys) == Set(fixture.templateArrays.keys))
      var expectedSavedBytes = 0
      for (key, original) in fixture.templateArrays {
        let actual = try #require(result[key])
        if key.contains(".gate.proj.") || key.contains(".embed_tokens.") {
          #expect(MLX.arrayEqual(actual, original).item(Bool.self))
        } else if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
          var expectedShape = original.shape
          expectedShape[expectedShape.count - 1] /= 2
          #expect(actual.shape == expectedShape)
          #expect(actual.dtype == original.dtype)
          expectedSavedBytes += original.nbytes / 2
        } else { #expect(actual.shape == original.shape && actual.dtype == original.dtype) }
      }
      let payload = result.values.reduce(0) { $0 + $1.nbytes }
      #expect(payload == fixture.templateArrays.values.reduce(0) { $0 + $1.nbytes } - expectedSavedBytes)
      let config = try #require(JSONSerialization.jsonObject(with: Data(contentsOf:
        fixture.output.appendingPathComponent("config.json"))) as? [String: Any])
      for key in ["quantization", "quantization_config"] {
        let policy = try #require(config[key] as? [String: Any])
        #expect(policy["group_size"] as? Int == 128)
        let router = try #require(policy[Fixture.prefix + "gate.proj"] as? [String: Any])
        #expect(router["group_size"] as? Int == 64 && router["bits"] as? Int == 8)
        let embedding = try #require(policy["language_model.model.embed_tokens"] as? [String: Any])
        #expect(embedding["group_size"] as? Int == 64 && embedding["bits"] as? Int == 4)
      }
      let outputIndex = try #require(JSONSerialization.jsonObject(with: Data(contentsOf:
        fixture.output.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any])
      #expect((outputIndex["metadata"] as? [String: Any])?["total_size"] as? Int == payload)
      let templateIndex = try #require(JSONSerialization.jsonObject(with: Data(contentsOf:
        fixture.template.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any])
      #expect(outputIndex["weight_map"] as? [String: String] == templateIndex["weight_map"] as? [String: String])
      let provenance = try #require(JSONSerialization.jsonObject(with: Data(contentsOf:
        fixture.output.appendingPathComponent("q4r8-scale-search.json"))) as? [String: Any])
      #expect(provenance["group_size"] as? Int == 128)
      #expect(provenance["output_tensor_bytes"] as? Int == payload)
      #expect(provenance["source_tensors_remaining"] as? Int == 0)
      #expect(provenance["algorithm"] as? String == (standard ? "q4r8_affine_standard" : "q4r8_affine_scale_search_ls2"))

      let source = try MLX.loadArrays(url: fixture.source.appendingPathComponent("model.safetensors"), stream: .cpu)
      let original = try #require(source["model.layers.1.mlp.experts.0.gate_proj.weight"])
      let native = MLX.quantized(original, groupSize: 128, bits: 4, mode: .affine)
      let nativeBiases = try #require(native.biases)
      let module = Fixture.switchPrefix + "gate_proj"
      let weight = try #require(result[module + ".weight"])[0]
      let scales = try #require(result[module + ".scales"])[0]
      let biases = try #require(result[module + ".biases"])[0]
      if standard {
        #expect(MLX.arrayEqual(weight, native.wq).item(Bool.self))
        #expect(MLX.arrayEqual(scales, native.scales).item(Bool.self))
        #expect(MLX.arrayEqual(biases, nativeBiases).item(Bool.self))
      } else {
        let candidate = MLX.dequantized(weight, scales: scales.asType(.float32), biases: biases.asType(.float32), groupSize: 128, bits: 4)
        let baseline = MLX.dequantized(native.wq, scales: native.scales.asType(.float32), biases: nativeBiases.asType(.float32), groupSize: 128, bits: 4)
        let candidateMSE = MLX.mean(MLX.square(candidate - original.asType(.float32))).item(Float.self)
        let baselineMSE = MLX.mean(MLX.square(baseline - original.asType(.float32))).item(Float.self)
        #expect(candidateMSE <= baselineMSE + 1e-7)
      }
    }
  }

  @Test("Group changes reject group-64 activation fallback before destination creation")
  func group128RejectsIncompatibleFallback() throws {
    let fixture = try Fixture(splitTemplate: true, inputWidth: 128)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    #expect(throws: (any Error).self) {
      try LagunaScaleSearchRescorer.rescore(source: fixture.source.path, template: fixture.template.path,
        destination: fixture.output.path, groupSize: 128, activationStats: fixture.calibration.path, cpu: true)
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
  }

  @Test("GPU-converted split templates pass strict GPU identity checks including preserved routers")
  func selectedDeviceIdentity() throws {
    let fixture = try Fixture(splitTemplate: true, quantizationOnGPU: true)
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try Device.withDefaultDevice(.gpu) {
      try LagunaScaleSearchRescorer.rescore(source: fixture.source.path, template: fixture.template.path,
        destination: fixture.output.path, activationStats: fixture.calibration.path,
        validationStats: fixture.validation.path, preflightOnly: true)
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
  }

  @Test("Source identity and corpus isolation fail before destination creation")
  func sourceAndCorpusValidation() throws {
    let fixture = try Fixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    #expect(throws: LagunaActivationInputError.self) {
      try LagunaScaleSearchRescorer.rescore(
        source: fixture.source.path, template: fixture.template.path, destination: fixture.output.path,
        activationStats: fixture.calibration.path, validationStats: fixture.calibration.path,
        preflightOnly: true, cpu: true)
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
    #expect(throws: LagunaActivationInputError.self) {
      try LagunaActivationStatistics.load(from: fixture.calibration, sourceURL: fixture.source,
        sourceConfig: Data("changed-source".utf8), sourceIndex: fixture.sourceIndexData)
    }
  }

  @Test("Changed source payload is rejected even when path, config, index, and file size are unchanged")
  func rejectsChangedWeightPayload() throws {
    let fixture = try Fixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let config = try Data(contentsOf: fixture.source.appendingPathComponent("config.json"))
    let weights = fixture.source.appendingPathComponent("model.safetensors")
    let originalSize = try weights.resourceValues(forKeys: [.fileSizeKey]).fileSize
    let file = try FileHandle(forUpdating: weights)
    let size = try file.seekToEnd()
    try file.seek(toOffset: size - 1)
    let byte = try #require(file.read(upToCount: 1)?.first)
    try file.seek(toOffset: size - 1)
    try file.write(contentsOf: Data([byte ^ 1]))
    try file.close()
    #expect(try weights.resourceValues(forKeys: [.fileSizeKey]).fileSize == originalSize)
    #expect(try Data(contentsOf: fixture.source.appendingPathComponent("model.safetensors.index.json"))
      == fixture.sourceIndexData)
    #expect(throws: LagunaActivationInputError.self) {
      try LagunaActivationStatistics.load(from: fixture.calibration, sourceURL: fixture.source,
        sourceConfig: config, sourceIndex: fixture.sourceIndexData)
    }
    #expect(!FileManager.default.fileExists(atPath: fixture.output.path))
  }

  @Test("Different aggregate hashes do not conceal partially overlapping calibration and dev segments")
  func rejectsPartialOverlap() throws {
    let fixture = try Fixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let config = try Data(contentsOf: fixture.source.appendingPathComponent("config.json"))
    let calibration = try LagunaActivationStatistics.load(from: fixture.calibration,
      sourceURL: fixture.source, sourceConfig: config, sourceIndex: fixture.sourceIndexData)
    var validation = try LagunaActivationStatistics.load(from: fixture.validation,
      sourceURL: fixture.source, sourceConfig: config, sourceIndex: fixture.sourceIndexData)
    validation.metadata["segment_token_fingerprints"] = "[\"other-segment\",\"calibration-segment\"]"
    #expect(throws: LagunaActivationInputError.self) { try validation.requireDisjoint(from: calibration) }
  }

  @Test("Source lifetimes preserve weights shared by fused and split output modules")
  func sharedSourceLifetimes() throws {
    let prefix = "language_model.model.layers.0.mlp."
    let sourcePrefix = "model.layers.0.mlp."
    let gate = sourcePrefix + "gate_proj.weight"
    let up = sourcePrefix + "up_proj.weight"
    let down = sourcePrefix + "down_proj.weight"
    let sourceKeys: Set<String> = [gate, up, down]
    let modules = ["gate_up_proj", "gate_proj", "up_proj", "down_proj"].map { prefix + $0 }
    let mapping = try Dictionary(uniqueKeysWithValues: modules.map {
      ($0, try LagunaScaleSearchRescorer.sourceKeys(for: $0, availableKeys: sourceKeys, numberOfExperts: 2))
    })
    var plan = LagunaSourceUsePlan(keysByModule: mapping)
    #expect(plan.remainingUses[gate] == 2)
    #expect(plan.remainingUses[up] == 2)
    #expect(try plan.finish(module: prefix + "gate_up_proj").isEmpty)
    #expect(plan.remainingUses[gate] == 1)
    #expect(plan.remainingUses[up] == 1)
    #expect(try plan.finish(module: prefix + "up_proj") == [up])
    #expect(plan.remainingUses[gate] == 1)
    #expect(try plan.finish(module: prefix + "gate_proj") == [gate])
    #expect(try plan.finish(module: prefix + "down_proj") == [down])
    #expect(plan.remainingUses.isEmpty)

    let routedPrefix = "model.layers.1.mlp.experts."
    let routedKeys = Set((0..<2).flatMap { expert in
      ["gate_proj", "up_proj", "down_proj"].map { "\(routedPrefix)\(expert).\($0).weight" }
    })
    let fusedKeys = try LagunaScaleSearchRescorer.sourceKeys(for: Fixture.switchPrefix + "gate_up_proj",
      availableKeys: routedKeys, numberOfExperts: 2)
    #expect(Set(fusedKeys) == Set(routedKeys.filter { !$0.hasSuffix("down_proj.weight") }))
    let downKeys = try LagunaScaleSearchRescorer.sourceKeys(for: Fixture.switchPrefix + "down_proj",
      availableKeys: routedKeys, numberOfExperts: 2)
    #expect(Set(downKeys) == Set(routedKeys.filter { $0.hasSuffix("down_proj.weight") }))
  }

  @Test("Fused dense input statistics map to legacy split gate/up template projections")
  func fusedDenseInputMapping() throws {
    let stats = LagunaActivationStatistics(
      moments: ["language_model.model.layers.0.mlp.gate_up_proj": MLXArray.ones([64])],
      counts: [:], metadata: [:], minimumExpertPositions: 32, url: URL(fileURLWithPath: "/unused"))
    #expect(try stats.momentPath(for: "language_model.model.layers.0.mlp.gate_proj")
      == "language_model.model.layers.0.mlp.gate_up_proj")
    #expect(try stats.momentPath(for: "language_model.model.layers.0.mlp.up_proj")
      == "language_model.model.layers.0.mlp.gate_up_proj")
    #expect(throws: LagunaActivationInputError.self) { try stats.momentPath(for: "unknown.down_proj") }
  }
}

private struct Fixture {
  static let prefix = "language_model.model.layers.1.mlp."
  static let switchPrefix = prefix + "switch_mlp."
  let root: URL
  let source: URL
  let template: URL
  let output: URL
  let calibration: URL
  let validation: URL
  let sourceIndexData: Data
  let templateArrays: [String: MLXArray]

  init(splitTemplate: Bool = false, quantizationOnGPU: Bool = false, inputWidth: Int = 64) throws {
    root = FileManager.default.temporaryDirectory.appendingPathComponent("laguna-awss-\(UUID().uuidString)")
    source = root.appendingPathComponent("source")
    template = root.appendingPathComponent("template")
    output = root.appendingPathComponent("output")
    calibration = root.appendingPathComponent("calibration.safetensors")
    validation = root.appendingPathComponent("dev.safetensors")
    for directory in [source, template] {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    let config = try JSONSerialization.data(withJSONObject: ["model_type": "laguna", "num_experts": 2], options: .sortedKeys)
    try config.write(to: source.appendingPathComponent("config.json"))
    let quantization: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine",
      Self.prefix + "gate.proj": ["group_size": 64, "bits": 8, "mode": "affine"]]
    try JSONSerialization.data(withJSONObject: ["model_type": "laguna", "num_experts": 2,
      "quantization": quantization, "quantization_config": quantization], options: .sortedKeys)
      .write(to: template.appendingPathComponent("config.json"))
    func weight(_ seed: Int, rows: Int? = nil) -> MLXArray {
      let rows = rows ?? inputWidth
      let values = (0..<(rows * inputWidth)).map { Float(sin(Double($0 + seed) * 0.137) + cos(Double($0) * 0.073)) }
      return MLXArray(values).reshaped(rows, inputWidth).asType(.bfloat16)
    }
    var original = [String: MLXArray]()
    original["model.layers.1.mlp.shared_expert.down_proj.weight"] = weight(11)
    original["model.layers.1.mlp.gate.weight"] = weight(29, rows: 2)
    original["model.embed_tokens.weight"] = weight(47, rows: 32)
    for expert in 0..<2 {
      for (index, projection) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
        original["model.layers.1.mlp.experts.\(expert).\(projection).weight"] = weight(61 + expert * 13 + index * 7)
      }
    }
    let additionalDirectModules = splitTemplate ? [
      "language_model.model.layers.0.mlp.gate_proj",
      "language_model.model.layers.0.mlp.up_proj",
      Self.prefix + "shared_expert.gate_proj",
      Self.prefix + "shared_expert.up_proj",
      "language_model.model.layers.0.self_attn.q_proj",
      "language_model.lm_head",
    ] : []
    for (index, module) in additionalDirectModules.enumerated() {
      original[String(module.dropFirst("language_model.".count)) + ".weight"] = weight(107 + index * 11)
    }
    var packed = [String: MLXArray]()
    func add(_ module: String, _ value: MLXArray, bits: Int = 4) throws {
      let result = MLX.quantized(value, groupSize: 64, bits: bits, mode: .affine,
        stream: quantizationOnGPU ? .gpu : .cpu)
      packed[module + ".weight"] = result.wq
      packed[module + ".scales"] = result.scales
      packed[module + ".biases"] = result.biases
    }
    try add(Self.prefix + "shared_expert.down_proj", original["model.layers.1.mlp.shared_expert.down_proj.weight"]!)
    try add(Self.prefix + "gate.proj", original["model.layers.1.mlp.gate.weight"]!, bits: 8)
    try add("language_model.model.embed_tokens", original["model.embed_tokens.weight"]!)
    for module in additionalDirectModules {
      try add(module, original[String(module.dropFirst("language_model.".count)) + ".weight"]!)
    }
    if splitTemplate {
      for projection in ["gate_proj", "up_proj"] {
        try add(Self.switchPrefix + projection, MLX.stacked((0..<2).map {
          original["model.layers.1.mlp.experts.\($0).\(projection).weight"]!
        }))
      }
    } else {
      let fused = (0..<2).map { expert in
        MLX.concatenated([original["model.layers.1.mlp.experts.\(expert).gate_proj.weight"]!,
          original["model.layers.1.mlp.experts.\(expert).up_proj.weight"]!], axis: -2)
      }
      try add(Self.switchPrefix + "gate_up_proj", MLX.stacked(fused))
    }
    try add(Self.switchPrefix + "down_proj", MLX.stacked((0..<2).map {
      original["model.layers.1.mlp.experts.\($0).down_proj.weight"]!
    }))
    templateArrays = packed
    sourceIndexData = try JSONSerialization.data(withJSONObject:
      ["weight_map": Dictionary(uniqueKeysWithValues: original.keys.map { ($0, "model.safetensors") })], options: .sortedKeys)
    let templateIndexData = try JSONSerialization.data(withJSONObject:
      ["weight_map": Dictionary(uniqueKeysWithValues: packed.keys.map { ($0, "model.safetensors") })], options: .sortedKeys)
    try MLX.save(arrays: original, url: source.appendingPathComponent("model.safetensors"))
    try MLX.save(arrays: packed, url: template.appendingPathComponent("model.safetensors"))
    try sourceIndexData.write(to: source.appendingPathComponent("model.safetensors.index.json"))
    try templateIndexData.write(to: template.appendingPathComponent("model.safetensors.index.json"))
    var statistics: [String: MLXArray] = [:]
    for module in [Self.prefix + "shared_expert.down_proj", Self.prefix + "gate.proj"] {
      statistics[module + LagunaActivationStatistics.momentSuffix] = MLXArray.ones([inputWidth], type: Float.self)
    }
    for module in additionalDirectModules {
      let path: String
      if module.hasSuffix(".gate_proj") || module.hasSuffix(".up_proj") {
        path = module.prefix(upTo: module.lastIndex(of: ".")!) + ".gate_up_proj"
      } else { path = module }
      statistics[path + LagunaActivationStatistics.momentSuffix] = MLXArray.ones([inputWidth], type: Float.self)
    }
    for module in [Self.switchPrefix + "gate_up_proj", Self.switchPrefix + "down_proj"] {
      statistics[module + LagunaActivationStatistics.momentSuffix] = MLXArray.ones([2, inputWidth], type: Float.self)
      statistics[module + LagunaActivationStatistics.countSuffix] = MLXArray([Int32(100), 1])
    }
    var metadata = [
      "format": "laguna_expert_activation_stats_v1",
      "algorithm": "expert_conditional_input_channel_second_moment", "model_type": "laguna",
      "dtype": "float32", "add_special_tokens": "true", "source_model": source.path,
      "source_config_fingerprint": LagunaActivationStatistics.fingerprint(config),
      "source_index_fingerprint": LagunaActivationStatistics.fingerprint(sourceIndexData),
      "source_weight_fingerprint_method": IndexedSafetensorsFingerprint.method,
      "source_weight_fingerprint": try IndexedSafetensorsFingerprint.compute(directory: source),
      "corpus_fingerprint": "calibration", "token_id_fingerprint": "calibration-tokens",
      "sample_token_fingerprints": "[\"calibration-sample\"]",
      "segment_token_fingerprints": "[\"calibration-segment\"]",
      "observed_token_count": "128", "module_count": String(statistics.values.filter { $0.ndim == 1 && $0.dtype == .float32 }.count), "expert_module_count": "2",
      "minimum_expert_positions": "32", "insufficient_coverage_policy": "retain_template_expert",
      "router_weighting": "conditional_on_selection_no_routing_score_weight",
    ]
    try MLX.save(arrays: statistics, metadata: metadata, url: calibration)
    metadata["corpus_fingerprint"] = "dev"
    metadata["token_id_fingerprint"] = "dev-tokens"
    metadata["sample_token_fingerprints"] = "[\"dev-sample\"]"
    metadata["segment_token_fingerprints"] = "[\"dev-segment\"]"
    try MLX.save(arrays: statistics, metadata: metadata, url: validation)
  }
}

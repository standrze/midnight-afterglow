import ArgumentParser
import Foundation
import MLX
import MistralActivationScaleSearchCore

private struct SafetensorsIndex: Decodable {
  var weightMap: [String: String]

  enum CodingKeys: String, CodingKey {
    case weightMap = "weight_map"
  }
}

private struct TemplateScaleSearchProvenance: Decodable {
  var algorithm: String
  var sourceModel: String
  var modelType: String
  var bits: Int
  var groupSize: Int
  var q4ScaleSearchModules: [String]
  var standardQ4Modules: [String]
  var q8Modules: [String]
  var skippedModules: [String]

  enum CodingKeys: String, CodingKey {
    case algorithm, bits
    case sourceModel = "source_model"
    case modelType = "model_type"
    case groupSize = "group_size"
    case q4ScaleSearchModules = "q4_scale_search_modules"
    case standardQ4Modules = "standard_q4_modules"
    case q8Modules = "q8_modules"
    case skippedModules = "skipped_modules"
  }
}

private struct TensorDescriptor {
  var shape: [Int]
  var dtype: DType
  var bytes: Int
}

private struct ActivationStatistics {
  static let tensorSuffix = ".input_second_moment"

  var url: URL
  var arraysByModule: [String: MLXArray]
  var metadata: [String: String]

  var corpusFingerprint: String { metadata["corpus_fingerprint"] ?? "" }
  var tokenIDFingerprint: String { metadata["token_id_fingerprint"] ?? "" }
  var observedTokenCount: Int { Int(metadata["observed_token_count"] ?? "") ?? 0 }
}

private struct AggregateDiagnostics: Encodable {
  var elementCount: Int
  var groupCount: Int
  var changedGroupCount: Int
  var validationRejectedGroupCount: Int
  var templateCalibrationWeightedMSE: Double
  var candidateCalibrationWeightedMSE: Double
  var calibrationWeightedMSEReductionPercent: Double
  var templateValidationWeightedMSE: Double?
  var candidateValidationWeightedMSE: Double?
  var validationWeightedMSEReductionPercent: Double?
  var templateRawMSE: Double
  var candidateRawMSE: Double
  var rawMSEReductionPercent: Double

  enum CodingKeys: String, CodingKey {
    case elementCount = "element_count"
    case groupCount = "group_count"
    case changedGroupCount = "changed_group_count"
    case validationRejectedGroupCount = "validation_rejected_group_count"
    case templateCalibrationWeightedMSE = "template_calibration_weighted_mse"
    case candidateCalibrationWeightedMSE = "candidate_calibration_weighted_mse"
    case calibrationWeightedMSEReductionPercent =
      "calibration_weighted_mse_reduction_percent"
    case templateValidationWeightedMSE = "template_validation_weighted_mse"
    case candidateValidationWeightedMSE = "candidate_validation_weighted_mse"
    case validationWeightedMSEReductionPercent =
      "validation_weighted_mse_reduction_percent"
    case templateRawMSE = "template_raw_mse"
    case candidateRawMSE = "candidate_raw_mse"
    case rawMSEReductionPercent = "raw_mse_reduction_percent"
  }
}

private struct RescoreProvenance: Encodable {
  var format = 1
  var status = "experimental_measured_candidate"
  var algorithm = "mistral_q4_affine_activation_weighted_scale_search_ls2_second_pass"
  var objective = "group_normalized_diagonal_expected_linear_output_squared_error_proxy"
  var objectiveNormalization = "each_64_input_channel_group_has_mean_weight_one"
  var createdAt: String
  var sourceModel: String
  var templateModel: String
  var calibrationStatistics: String
  var validationStatistics: String?
  var sourceModelType: String
  var bits = 4
  var groupSize = 64
  var searchFactors = MistralActivationWeightedScaleSearch.searchFactors.map(Double.init)
  var weightedBiasRefinementIterations = 1
  var weightedJointAffineRefinementIterations = 2
  var validationGuard = false
  var calibrationCorpusFingerprint: String
  var calibrationTokenIDFingerprint: String
  var calibrationObservedTokenCount: Int
  var validationCorpusFingerprint: String?
  var validationTokenIDFingerprint: String?
  var validationObservedTokenCount: Int?
  var q4ModulesRescored: Int
  var preservedStandardQ4Modules: [String]
  var preservedQ8Modules: [String]
  var preservedSkippedModules: [String]
  var aggregate: AggregateDiagnostics
  var modules: [String: ActivationWeightedScaleSearchDiagnostics]

  enum CodingKeys: String, CodingKey {
    case format, status, algorithm, objective, bits, aggregate, modules
    case objectiveNormalization = "objective_normalization"
    case createdAt = "created_at"
    case sourceModel = "source_model"
    case templateModel = "template_model"
    case calibrationStatistics = "calibration_statistics"
    case validationStatistics = "validation_statistics"
    case sourceModelType = "source_model_type"
    case groupSize = "group_size"
    case searchFactors = "search_factors"
    case weightedBiasRefinementIterations = "weighted_bias_refinement_iterations"
    case weightedJointAffineRefinementIterations =
      "weighted_joint_affine_refinement_iterations"
    case validationGuard = "validation_guard"
    case calibrationCorpusFingerprint = "calibration_corpus_fingerprint"
    case calibrationTokenIDFingerprint = "calibration_token_id_fingerprint"
    case calibrationObservedTokenCount = "calibration_observed_token_count"
    case validationCorpusFingerprint = "validation_corpus_fingerprint"
    case validationTokenIDFingerprint = "validation_token_id_fingerprint"
    case validationObservedTokenCount = "validation_observed_token_count"
    case q4ModulesRescored = "q4_modules_rescored"
    case preservedStandardQ4Modules = "preserved_standard_q4_modules"
    case preservedQ8Modules = "preserved_q8_modules"
    case preservedSkippedModules = "preserved_skipped_modules"
  }
}

private enum RescoreError: Error, LocalizedError {
  case invalidInput(String)
  case incompatibleTemplate(String)
  case invalidStatistics(String)
  case missingTensor(String)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let message): "Invalid activation rescore input: \(message)"
    case .incompatibleTemplate(let message): "Incompatible Mistral LS2 template: \(message)"
    case .invalidStatistics(let message): "Invalid activation statistics: \(message)"
    case .missingTensor(let key): "Missing tensor: \(key)"
    }
  }
}

private final class SourceShardCache {
  let directory: URL
  let index: SafetensorsIndex
  private var shardName: String?
  private var arrays = [String: MLXArray]()

  init(directory: URL, index: SafetensorsIndex) {
    self.directory = directory
    self.index = index
  }

  func tensor(_ key: String) throws -> MLXArray {
    guard let requestedShard = index.weightMap[key] else {
      throw RescoreError.missingTensor(key)
    }
    if requestedShard != shardName {
      arrays.removeAll(keepingCapacity: false)
      Memory.clearCache()
      arrays = try loadArrays(
        url: directory.appendingPathComponent(requestedShard), stream: .cpu)
      shardName = requestedShard
    }
    guard let value = arrays[key] else {
      throw RescoreError.missingTensor("\(key) in declared shard \(requestedShard)")
    }
    return value
  }

  func clear() {
    arrays.removeAll(keepingCapacity: false)
    shardName = nil
    Memory.clearCache()
  }
}

@main
private struct MistralActivationScaleSearchRescorer: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: CommandLine.arguments.first?.split(separator: "/").last.map(String.init)
      ?? "wick-mistral-awss-quantize",
    abstract:
      "Refine dense Mistral LS2 Q4 grids using BF16 input-channel activation second moments."
  )

  @Argument(help: "Original unquantized dense Mistral safetensors directory.")
  var source: String

  @Argument(help: "Read-only affine-Q4 ScaleSearch LS2 checkpoint used as the exact template.")
  var template: String

  @Argument(help: "Calibration activation-statistics safetensors file.")
  var calibrationStats: String

  @Argument(help: "New destination checkpoint directory; it must not already exist.")
  var destination: String

  @Option(
    name: .customLong("validation-stats"),
    help:
      "Optional disjoint activation-statistics file; candidate groups must not worsen its weighted error."
  )
  var validationStats: String?

  @Flag(
    help: "Validate every mapping, shape, dtype, fingerprint, and statistic without writing output."
  )
  var preflightOnly = false

  @Flag(help: "Run activation-weighted search on CPU rather than the default accelerator.")
  var cpu = false

  mutating func run() throws {
    #if os(Linux)
      defer { clearStreams() }
    #endif

    if cpu {
      try Device.withDefaultDevice(.cpu) { try execute() }
    } else {
      try execute()
    }
  }

  private func execute() throws {
    Memory.cacheLimit = 512 * 1_024 * 1_024

    let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
    let templateURL = URL(fileURLWithPath: template).standardizedFileURL
    let calibrationURL = URL(fileURLWithPath: calibrationStats).standardizedFileURL
    let validationURL = validationStats.map {
      URL(fileURLWithPath: $0).standardizedFileURL
    }
    let destinationURL = URL(fileURLWithPath: destination).standardizedFileURL
    try validateDirectory(sourceURL, label: "source")
    try validateDirectory(templateURL, label: "template")
    try validateFile(calibrationURL, label: "calibration statistics")
    if let validationURL { try validateFile(validationURL, label: "validation statistics") }
    let distinctURLs =
      [sourceURL, templateURL, calibrationURL, destinationURL] + (validationURL.map { [$0] } ?? [])
    guard Set(distinctURLs.map(\.path)).count == distinctURLs.count else {
      throw RescoreError.invalidInput(
        "source, template, statistics, and destination paths must be distinct")
    }
    guard !FileManager.default.fileExists(atPath: destinationURL.path) else {
      throw RescoreError.invalidInput("destination already exists: \(destinationURL.path)")
    }

    let sourceConfigURL = sourceURL.appendingPathComponent("config.json")
    let templateConfigURL = templateURL.appendingPathComponent("config.json")
    let sourceConfigData = try Data(contentsOf: sourceConfigURL)
    let templateConfigData = try Data(contentsOf: templateConfigURL)
    let sourceModelType = try validateDenseMistral(
      configData: sourceConfigData, label: "source")
    _ = try validateDenseMistral(configData: templateConfigData, label: "template")

    let sourceIndexURL = sourceURL.appendingPathComponent("model.safetensors.index.json")
    let templateIndexURL = templateURL.appendingPathComponent("model.safetensors.index.json")
    let sourceIndexData = try Data(contentsOf: sourceIndexURL)
    let templateIndexData = try Data(contentsOf: templateIndexURL)
    let sourceIndex = try JSONDecoder().decode(SafetensorsIndex.self, from: sourceIndexData)
    let templateIndex = try JSONDecoder().decode(SafetensorsIndex.self, from: templateIndexData)
    try rejectMoEWeights(sourceIndex.weightMap.keys, label: "source")
    try rejectMoEWeights(templateIndex.weightMap.keys, label: "template")

    let templateProvenanceURL = templateURL.appendingPathComponent(
      "scale-search-quantization.json")
    guard FileManager.default.fileExists(atPath: templateProvenanceURL.path) else {
      throw RescoreError.incompatibleTemplate(
        "missing scale-search-quantization.json; an ordinary Q4 model is not an LS2 baseline")
    }
    let templateProvenance = try JSONDecoder().decode(
      TemplateScaleSearchProvenance.self,
      from: Data(contentsOf: templateProvenanceURL)
    )
    guard templateProvenance.algorithm == "q4r8_affine_scale_search_ls2",
      templateProvenance.bits == 4,
      templateProvenance.groupSize == 64,
      isMistralType(templateProvenance.modelType),
      !templateProvenance.q4ScaleSearchModules.isEmpty
    else {
      throw RescoreError.incompatibleTemplate(
        "provenance must identify a Mistral q4r8_affine_scale_search_ls2 group-64 model")
    }
    let recordedSource = URL(fileURLWithPath: templateProvenance.sourceModel)
      .standardizedFileURL
    guard recordedSource == sourceURL else {
      throw RescoreError.incompatibleTemplate(
        "LS2 provenance names source \(recordedSource.path), not \(sourceURL.path)")
    }

    let sourceConfigFingerprint = fnv1a64(sourceConfigData)
    let sourceIndexFingerprint = fnv1a64(sourceIndexData)
    let calibration = try loadStatistics(
      calibrationURL,
      expectedModelType: sourceModelType,
      sourceConfigFingerprint: sourceConfigFingerprint,
      sourceIndexFingerprint: sourceIndexFingerprint,
      label: "calibration"
    )
    let validation = try validationURL.map {
      try loadStatistics(
        $0,
        expectedModelType: sourceModelType,
        sourceConfigFingerprint: sourceConfigFingerprint,
        sourceIndexFingerprint: sourceIndexFingerprint,
        label: "validation"
      )
    }
    if let validation {
      guard calibration.corpusFingerprint != validation.corpusFingerprint,
        calibration.tokenIDFingerprint != validation.tokenIDFingerprint
      else {
        throw RescoreError.invalidStatistics(
          "validation statistics must come from a disjoint fingerprinted corpus")
      }
    }

    let q4Modules = templateProvenance.q4ScaleSearchModules.sorted()
    guard Set(q4Modules).count == q4Modules.count else {
      throw RescoreError.incompatibleTemplate("q4_scale_search_modules contains duplicates")
    }
    try rejectMoEModules(q4Modules)
    guard Set(calibration.arraysByModule.keys) == Set(q4Modules) else {
      throw statisticsCoverageError(
        actual: Set(calibration.arraysByModule.keys), expected: Set(q4Modules),
        label: "calibration")
    }
    if let validation, Set(validation.arraysByModule.keys) != Set(q4Modules) {
      throw statisticsCoverageError(
        actual: Set(validation.arraysByModule.keys), expected: Set(q4Modules),
        label: "validation")
    }

    let sourceKeyByModule = try Dictionary(
      uniqueKeysWithValues: q4Modules.map { module in
        (module, try sourceWeightKey(for: module, sourceIndex: sourceIndex))
      }
    )
    let sourceDescriptors = try scanDescriptors(
      directory: sourceURL, index: sourceIndex, label: "source")
    let templateDescriptors = try scanDescriptors(
      directory: templateURL, index: templateIndex, label: "template")
    try validateModuleGeometry(
      modules: q4Modules,
      sourceKeyByModule: sourceKeyByModule,
      sourceDescriptors: sourceDescriptors,
      templateDescriptors: templateDescriptors,
      templateIndex: templateIndex,
      calibration: calibration,
      validation: validation
    )

    let crossShardModules = q4Modules.filter { module in
      Set(
        ["weight", "scales", "biases"].compactMap {
          templateIndex.weightMap["\(module).\($0)"]
        }
      ).count > 1
    }
    print(
      "Mistral activation-weighted LS2 preflight: \(q4Modules.count) dense Q4 modules, "
        + "\(crossShardModules.count) cross-shard module(s), "
        + "validation_guard=\(validation != nil), device=\(Device.defaultDevice())"
    )
    print(
      "calibration tokens=\(calibration.observedTokenCount) "
        + "corpus=\(calibration.corpusFingerprint) "
        + "template tensors=\(templateIndex.weightMap.count)"
    )
    if let validation {
      print(
        "validation tokens=\(validation.observedTokenCount) "
          + "corpus=\(validation.corpusFingerprint)")
    }
    if preflightOnly {
      print("Preflight passed; no destination was written.")
      return
    }

    let fileManager = FileManager.default
    let stagingURL = destinationURL.deletingLastPathComponent().appendingPathComponent(
      ".\(destinationURL.lastPathComponent).activation-rescore-\(UUID().uuidString)")
    try fileManager.createDirectory(at: stagingURL, withIntermediateDirectories: false)
    var committed = false
    defer {
      if !committed { try? fileManager.removeItem(at: stagingURL) }
    }
    try copySidecars(from: templateURL, to: stagingURL)

    let sourceCache = SourceShardCache(directory: sourceURL, index: sourceIndex)
    var diagnostics = [String: ActivationWeightedScaleSearchDiagnostics]()
    var crossShardReplacements = [String: MLXArray]()
    for (offset, module) in crossShardModules.enumerated() {
      let result = try rescoreModule(
        module,
        sourceWeight: sourceCache.tensor(sourceKeyByModule[module]!),
        templateWeight: try loadTensor(
          "\(module).weight", directory: templateURL, index: templateIndex),
        templateScales: try loadTensor(
          "\(module).scales", directory: templateURL, index: templateIndex),
        templateBiases: try loadTensor(
          "\(module).biases", directory: templateURL, index: templateIndex),
        calibration: calibration,
        validation: validation
      )
      diagnostics[module] = result.diagnostics
      try appendReplacement(result, module: module, to: &crossShardReplacements)
      print(moduleSummary(offset + 1, q4Modules.count, module, result.diagnostics))
      Memory.clearCache()
    }
    sourceCache.clear()

    let crossShardSet = Set(crossShardModules)
    var modulesByShard = [String: [String]]()
    for module in q4Modules where !crossShardSet.contains(module) {
      guard let shard = templateIndex.weightMap["\(module).weight"] else {
        throw RescoreError.incompatibleTemplate("missing \(module).weight placement")
      }
      modulesByShard[shard, default: []].append(module)
    }
    for shard in modulesByShard.keys {
      modulesByShard[shard]!.sort {
        let lhsShard = sourceIndex.weightMap[sourceKeyByModule[$0]!] ?? ""
        let rhsShard = sourceIndex.weightMap[sourceKeyByModule[$1]!] ?? ""
        return lhsShard == rhsShard ? $0 < $1 : lhsShard < rhsShard
      }
    }

    var completedModules = crossShardModules.count
    let shardNames = Set(templateIndex.weightMap.values).sorted()
    for (shardOffset, shardName) in shardNames.enumerated() {
      let templateShardURL = templateURL.appendingPathComponent(shardName)
      var (arrays, metadata) = try loadArraysAndMetadata(url: templateShardURL, stream: .cpu)
      let modules = modulesByShard[shardName] ?? []
      print(
        "[template shard \(shardOffset + 1)/\(shardNames.count)] \(shardName): "
          + "\(modules.count) local module(s)")

      for key in crossShardReplacements.keys.sorted()
      where templateIndex.weightMap[key] == shardName {
        guard let value = crossShardReplacements.removeValue(forKey: key) else { continue }
        try install(value, key: key, arrays: &arrays)
      }

      for module in modules {
        guard let templateWeight = arrays["\(module).weight"],
          let templateScales = arrays["\(module).scales"],
          let templateBiases = arrays["\(module).biases"]
        else {
          throw RescoreError.incompatibleTemplate(
            "same-shard module \(module) is missing one of weight/scales/biases")
        }
        let result = try rescoreModule(
          module,
          sourceWeight: sourceCache.tensor(sourceKeyByModule[module]!),
          templateWeight: templateWeight,
          templateScales: templateScales,
          templateBiases: templateBiases,
          calibration: calibration,
          validation: validation
        )
        try install(result.weight, key: "\(module).weight", arrays: &arrays)
        try install(result.scales, key: "\(module).scales", arrays: &arrays)
        try install(result.biases, key: "\(module).biases", arrays: &arrays)
        diagnostics[module] = result.diagnostics
        completedModules += 1
        print(moduleSummary(completedModules, q4Modules.count, module, result.diagnostics))
        Memory.clearCache()
      }

      Stream.defaultStream(Device.defaultDevice()).synchronize()
      let partialURL = stagingURL.appendingPathComponent(".\(shardName).partial.safetensors")
      try save(arrays: arrays, metadata: metadata, url: partialURL)
      try fileManager.moveItem(
        at: partialURL,
        to: stagingURL.appendingPathComponent(shardName)
      )
      arrays.removeAll(keepingCapacity: false)
      metadata.removeAll(keepingCapacity: false)
      Memory.clearCache()
    }
    sourceCache.clear()
    guard crossShardReplacements.isEmpty, diagnostics.count == q4Modules.count,
      completedModules == q4Modules.count
    else {
      throw RescoreError.incompatibleTemplate(
        "stream ended with \(crossShardReplacements.count) replacement tensor(s) and "
          + "\(completedModules)/\(q4Modules.count) modules")
    }

    try fileManager.copyItem(
      at: templateIndexURL,
      to: stagingURL.appendingPathComponent("model.safetensors.index.json")
    )
    let aggregate = aggregateDiagnostics(Array(diagnostics.values))
    let provenance = RescoreProvenance(
      createdAt: ISO8601DateFormatter().string(from: Date()),
      sourceModel: sourceURL.path,
      templateModel: templateURL.path,
      calibrationStatistics: calibrationURL.path,
      validationStatistics: validationURL?.path,
      sourceModelType: sourceModelType,
      validationGuard: validation != nil,
      calibrationCorpusFingerprint: calibration.corpusFingerprint,
      calibrationTokenIDFingerprint: calibration.tokenIDFingerprint,
      calibrationObservedTokenCount: calibration.observedTokenCount,
      validationCorpusFingerprint: validation?.corpusFingerprint,
      validationTokenIDFingerprint: validation?.tokenIDFingerprint,
      validationObservedTokenCount: validation?.observedTokenCount,
      q4ModulesRescored: q4Modules.count,
      preservedStandardQ4Modules: templateProvenance.standardQ4Modules.sorted(),
      preservedQ8Modules: templateProvenance.q8Modules.sorted(),
      preservedSkippedModules: templateProvenance.skippedModules.sorted(),
      aggregate: aggregate,
      modules: diagnostics
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(provenance).write(
      to: stagingURL.appendingPathComponent(
        "activation-scale-search-quantization.json"),
      options: .atomic
    )
    try fileManager.moveItem(at: stagingURL, to: destinationURL)
    committed = true
    print(
      String(
        format:
          "Created %@; calibration weighted MSE %.8g -> %.8g (%+.4f%%), changed %d/%d groups.",
        destinationURL.path,
        aggregate.templateCalibrationWeightedMSE,
        aggregate.candidateCalibrationWeightedMSE,
        aggregate.calibrationWeightedMSEReductionPercent,
        aggregate.changedGroupCount,
        aggregate.groupCount
      )
    )
  }

  private func rescoreModule(
    _ module: String,
    sourceWeight: MLXArray,
    templateWeight: MLXArray,
    templateScales: MLXArray,
    templateBiases: MLXArray,
    calibration: ActivationStatistics,
    validation: ActivationStatistics?
  ) throws -> ActivationWeightedScaleSearchResult {
    guard let calibrationMoments = calibration.arraysByModule[module] else {
      throw RescoreError.invalidStatistics("calibration is missing \(module)")
    }
    let result = try MistralActivationWeightedScaleSearch.rescore(
      sourceWeight: sourceWeight,
      templateWeight: templateWeight,
      templateScales: templateScales,
      templateBiases: templateBiases,
      calibrationSecondMoments: calibrationMoments,
      validationSecondMoments: validation?.arraysByModule[module]
    )
    MLX.eval(result.weight, result.scales, result.biases)
    Stream.defaultStream(Device.defaultDevice()).synchronize()
    return result
  }

  private func appendReplacement(
    _ result: ActivationWeightedScaleSearchResult,
    module: String,
    to replacements: inout [String: MLXArray]
  ) throws {
    for (key, value) in [
      ("\(module).weight", result.weight),
      ("\(module).scales", result.scales),
      ("\(module).biases", result.biases),
    ] {
      guard replacements.updateValue(value, forKey: key) == nil else {
        throw RescoreError.incompatibleTemplate("duplicate replacement \(key)")
      }
    }
  }

  private func install(
    _ value: MLXArray,
    key: String,
    arrays: inout [String: MLXArray]
  ) throws {
    guard let templateValue = arrays[key] else {
      throw RescoreError.incompatibleTemplate("\(key) is not in its declared shard")
    }
    guard value.shape == templateValue.shape,
      value.dtype == templateValue.dtype,
      value.nbytes == templateValue.nbytes
    else {
      throw RescoreError.incompatibleTemplate(
        "\(key) expected shape=\(templateValue.shape) dtype=\(templateValue.dtype) "
          + "bytes=\(templateValue.nbytes), got shape=\(value.shape) "
          + "dtype=\(value.dtype) bytes=\(value.nbytes)")
    }
    arrays[key] = value
  }

  private func validateModuleGeometry(
    modules: [String],
    sourceKeyByModule: [String: String],
    sourceDescriptors: [String: TensorDescriptor],
    templateDescriptors: [String: TensorDescriptor],
    templateIndex: SafetensorsIndex,
    calibration: ActivationStatistics,
    validation: ActivationStatistics?
  ) throws {
    for module in modules {
      guard let sourceKey = sourceKeyByModule[module],
        let source = sourceDescriptors[sourceKey]
      else { throw RescoreError.missingTensor(sourceKeyByModule[module] ?? module) }
      guard source.shape.count == 2, let inputWidth = source.shape.last,
        inputWidth > 0, inputWidth % 64 == 0,
        source.dtype.isFloatingPoint, !source.dtype.isComplex
      else {
        throw RescoreError.invalidInput(
          "\(module) must map to a dense real rank-2 group-64 source weight; got "
            + "\(source.shape) \(source.dtype)")
      }
      var expectedWeightShape = source.shape
      expectedWeightShape[expectedWeightShape.count - 1] = inputWidth / 8
      var expectedMetadataShape = source.shape
      expectedMetadataShape[expectedMetadataShape.count - 1] = inputWidth / 64
      guard let packed = templateDescriptors["\(module).weight"],
        let scales = templateDescriptors["\(module).scales"],
        let biases = templateDescriptors["\(module).biases"]
      else {
        throw RescoreError.incompatibleTemplate("missing affine triplet for \(module)")
      }
      guard packed.shape == expectedWeightShape, packed.dtype == .uint32,
        scales.shape == expectedMetadataShape, biases.shape == expectedMetadataShape,
        scales.dtype == source.dtype, biases.dtype == source.dtype,
        templateIndex.weightMap["\(module).weight"] != nil,
        templateIndex.weightMap["\(module).scales"] != nil,
        templateIndex.weightMap["\(module).biases"] != nil
      else {
        throw RescoreError.incompatibleTemplate(
          "\(module) does not retain source-compatible affine Q4 group-64 geometry")
      }
      try validateMomentArray(
        calibration.arraysByModule[module], module: module,
        inputWidth: inputWidth, label: "calibration")
      if let validation {
        try validateMomentArray(
          validation.arraysByModule[module], module: module,
          inputWidth: inputWidth, label: "validation")
      }
    }
  }

  private func validateMomentArray(
    _ moments: MLXArray?,
    module: String,
    inputWidth: Int,
    label: String
  ) throws {
    guard let moments, moments.shape == [inputWidth], moments.dtype == .float32 else {
      throw RescoreError.invalidStatistics(
        "\(label) \(module) must be rank-1 Float32 [\(inputWidth)]")
    }
    let values = moments.asArray(Float.self)
    for start in stride(from: 0, to: inputWidth, by: 64) {
      var groupSum: Double = 0
      for offset in 0..<64 {
        let value = values[start + offset]
        guard value.isFinite, value >= 0 else {
          throw RescoreError.invalidStatistics(
            "\(label) \(module) channel \(start + offset) is not finite/nonnegative")
        }
        groupSum += Double(value)
      }
      guard groupSum.isFinite, groupSum > 0 else {
        throw RescoreError.invalidStatistics(
          "\(label) \(module) group \(start / 64) has no positive activation mass")
      }
    }
  }

  private func loadStatistics(
    _ url: URL,
    expectedModelType: String,
    sourceConfigFingerprint: String,
    sourceIndexFingerprint: String,
    label: String
  ) throws -> ActivationStatistics {
    let (arrays, metadata) = try loadArraysAndMetadata(url: url, stream: .cpu)
    guard metadata["format"] == "mistral_activation_stats_v1",
      metadata["algorithm"] == "input_channel_second_moment",
      metadata["dtype"] == "float32",
      metadata["add_special_tokens"] == "true",
      let modelType = metadata["model_type"], isMistralType(modelType),
      isMistralType(expectedModelType),
      modelType.lowercased() == expectedModelType.lowercased(),
      let sourceModel = metadata["source_model"], !sourceModel.isEmpty,
      metadata["source_config_fingerprint"] == sourceConfigFingerprint,
      metadata["source_index_fingerprint"] == sourceIndexFingerprint,
      let corpusFingerprint = metadata["corpus_fingerprint"], !corpusFingerprint.isEmpty,
      let tokenFingerprint = metadata["token_id_fingerprint"], !tokenFingerprint.isEmpty,
      let observed = metadata["observed_token_count"].flatMap(Int.init), observed > 0,
      let moduleCount = metadata["module_count"].flatMap(Int.init), moduleCount > 0,
      let segmentLimit = metadata["segment_token_limit"].flatMap(Int.init), segmentLimit > 0
    else {
      throw RescoreError.invalidStatistics(
        "\(label) metadata is incomplete, incompatible, or does not match the source")
    }
    var byModule = [String: MLXArray]()
    for (key, value) in arrays {
      guard key.hasSuffix(ActivationStatistics.tensorSuffix) else {
        throw RescoreError.invalidStatistics(
          "\(label) tensor \(key) lacks suffix \(ActivationStatistics.tensorSuffix)")
      }
      let module = String(key.dropLast(ActivationStatistics.tensorSuffix.count))
      guard !module.isEmpty, byModule.updateValue(value, forKey: module) == nil else {
        throw RescoreError.invalidStatistics(
          "\(label) contains an empty or duplicate module path for \(key)")
      }
    }
    guard byModule.count == moduleCount else {
      throw RescoreError.invalidStatistics(
        "\(label) metadata declares \(moduleCount) modules but file contains \(byModule.count)")
    }
    return ActivationStatistics(url: url, arraysByModule: byModule, metadata: metadata)
  }

  private func scanDescriptors(
    directory: URL,
    index: SafetensorsIndex,
    label: String
  ) throws -> [String: TensorDescriptor] {
    var result = [String: TensorDescriptor]()
    let keysByShard = Dictionary(grouping: index.weightMap.keys) {
      index.weightMap[$0]!
    }
    for shardName in keysByShard.keys.sorted() {
      let arrays = try loadArrays(
        url: directory.appendingPathComponent(shardName), stream: .cpu)
      for key in keysByShard[shardName] ?? [] {
        guard let value = arrays[key] else {
          throw RescoreError.missingTensor("\(label) \(key) in \(shardName)")
        }
        guard
          result.updateValue(
            TensorDescriptor(shape: value.shape, dtype: value.dtype, bytes: value.nbytes),
            forKey: key
          ) == nil
        else {
          throw RescoreError.invalidInput("\(label) index repeats tensor \(key)")
        }
      }
      Memory.clearCache()
    }
    guard result.count == index.weightMap.count else {
      throw RescoreError.invalidInput(
        "\(label) index maps \(index.weightMap.count) tensors but scanned \(result.count)")
    }
    return result
  }

  private func sourceWeightKey(
    for module: String,
    sourceIndex: SafetensorsIndex
  ) throws -> String {
    let candidates = ["language_model.\(module).weight", "\(module).weight"]
    let matches = candidates.filter { sourceIndex.weightMap[$0] != nil }
    guard matches.count == 1, let match = matches.first else {
      throw RescoreError.invalidInput(
        "\(module) must map to exactly one source tensor; tried \(candidates.joined(separator: ", "))"
      )
    }
    return match
  }

  private func loadTensor(
    _ key: String,
    directory: URL,
    index: SafetensorsIndex
  ) throws -> MLXArray {
    guard let shard = index.weightMap[key] else { throw RescoreError.missingTensor(key) }
    let arrays = try loadArrays(
      url: directory.appendingPathComponent(shard), stream: .cpu)
    guard let value = arrays[key] else {
      throw RescoreError.missingTensor("\(key) in declared shard \(shard)")
    }
    return value
  }

  private func copySidecars(from template: URL, to destination: URL) throws {
    for item in try FileManager.default.contentsOfDirectory(
      at: template,
      includingPropertiesForKeys: [.isDirectoryKey],
      options: [.skipsHiddenFiles]
    ) {
      let values = try item.resourceValues(forKeys: [.isDirectoryKey])
      if values.isDirectory == true { continue }
      if item.pathExtension == "safetensors"
        || item.lastPathComponent == "model.safetensors.index.json"
        || item.lastPathComponent == "scale-search-quantization.json"
        || item.lastPathComponent == "activation-scale-search-quantization.json"
      {
        continue
      }
      try FileManager.default.copyItem(
        at: item,
        to: destination.appendingPathComponent(item.lastPathComponent)
      )
    }
  }

  private func validateDenseMistral(configData: Data, label: String) throws -> String {
    guard let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
      throw RescoreError.invalidInput("\(label) config.json is not an object")
    }
    let outerType = root["model_type"] as? String
    let textConfig = root["text_config"] as? [String: Any]
    let innerType = textConfig?["model_type"] as? String
    guard [outerType, innerType].compactMap({ $0 }).contains(where: isMistralType) else {
      throw RescoreError.invalidInput(
        "\(label) is not Mistral/Ministral (model_type=\(outerType ?? "missing"), "
          + "text model_type=\(innerType ?? "missing"))")
    }
    if containsPositiveExpertCount(root) {
      throw RescoreError.invalidInput("\(label) declares a mixture-of-experts configuration")
    }
    return outerType ?? innerType!
  }

  private func containsPositiveExpertCount(_ value: Any) -> Bool {
    if let dictionary = value as? [String: Any] {
      let expertKeys: Set<String> = [
        "num_experts", "num_local_experts", "n_routed_experts", "num_experts_per_tok",
      ]
      for (key, child) in dictionary {
        if expertKeys.contains(key), let number = child as? NSNumber, number.intValue > 0 {
          return true
        }
        if containsPositiveExpertCount(child) { return true }
      }
    } else if let array = value as? [Any] {
      return array.contains { containsPositiveExpertCount($0) }
    }
    return false
  }

  private func rejectMoEWeights(_ keys: Dictionary<String, String>.Keys, label: String) throws {
    let marker = keys.first(where: isMoEPath)
    if let marker {
      throw RescoreError.invalidInput("\(label) contains routed/MoE tensor \(marker)")
    }
  }

  private func rejectMoEModules(_ modules: [String]) throws {
    if let module = modules.first(where: isMoEPath) {
      throw RescoreError.incompatibleTemplate("routed/MoE module is unsupported: \(module)")
    }
  }

  private func isMoEPath(_ path: String) -> Bool {
    path.contains(".experts.")
      || path.contains(".switch_mlp.")
      || path.contains(".block_sparse_moe.")
      || path.contains(".router.")
      || path.hasSuffix(".router.weight")
  }

  private func isMistralType(_ value: String) -> Bool {
    ["mistral", "mistral3", "ministral3"].contains(value.lowercased())
  }

  private func statisticsCoverageError(
    actual: Set<String>, expected: Set<String>, label: String
  ) -> RescoreError {
    let missing = expected.subtracting(actual).sorted()
    let extra = actual.subtracting(expected).sorted()
    return .invalidStatistics(
      "\(label) module coverage differs from LS2 Q4 modules; missing="
        + "\(missing.prefix(8).joined(separator: ",")) extra="
        + "\(extra.prefix(8).joined(separator: ","))")
  }

  private func aggregateDiagnostics(
    _ values: [ActivationWeightedScaleSearchDiagnostics]
  ) -> AggregateDiagnostics {
    let elementCount = values.reduce(0) { $0 + $1.elementCount }
    let weighted: (KeyPath<ActivationWeightedScaleSearchDiagnostics, Double>) -> Double = {
      keyPath in
      guard elementCount > 0 else { return 0 }
      return values.reduce(0) {
        $0 + $1[keyPath: keyPath] * Double($1.elementCount)
      } / Double(elementCount)
    }
    let templateCalibration = weighted(\.templateCalibrationWeightedMSE)
    let candidateCalibration = weighted(\.candidateCalibrationWeightedMSE)
    let templateRaw = weighted(\.templateRawMSE)
    let candidateRaw = weighted(\.candidateRawMSE)
    let hasValidation = values.allSatisfy { $0.templateValidationWeightedMSE != nil }
    let templateValidation: Double? =
      hasValidation && elementCount > 0
      ? values.reduce(0) {
        $0 + $1.templateValidationWeightedMSE! * Double($1.elementCount)
      } / Double(elementCount)
      : nil
    let candidateValidation: Double? =
      hasValidation && elementCount > 0
      ? values.reduce(0) {
        $0 + $1.candidateValidationWeightedMSE! * Double($1.elementCount)
      } / Double(elementCount)
      : nil
    return AggregateDiagnostics(
      elementCount: elementCount,
      groupCount: values.reduce(0) { $0 + $1.groupCount },
      changedGroupCount: values.reduce(0) { $0 + $1.changedGroupCount },
      validationRejectedGroupCount: values.reduce(0) {
        $0 + $1.validationRejectedGroupCount
      },
      templateCalibrationWeightedMSE: templateCalibration,
      candidateCalibrationWeightedMSE: candidateCalibration,
      calibrationWeightedMSEReductionPercent: percentReduction(
        from: templateCalibration, to: candidateCalibration),
      templateValidationWeightedMSE: templateValidation,
      candidateValidationWeightedMSE: candidateValidation,
      validationWeightedMSEReductionPercent: optionalPercentReduction(
        from: templateValidation, to: candidateValidation),
      templateRawMSE: templateRaw,
      candidateRawMSE: candidateRaw,
      rawMSEReductionPercent: percentReduction(from: templateRaw, to: candidateRaw)
    )
  }

  private func optionalPercentReduction(from: Double?, to: Double?) -> Double? {
    guard let from, let to else { return nil }
    return percentReduction(from: from, to: to)
  }

  private func percentReduction(from: Double, to: Double) -> Double {
    guard from.isFinite, to.isFinite, from > 0 else { return 0 }
    return (from - to) / from * 100
  }

  private func moduleSummary(
    _ completed: Int,
    _ total: Int,
    _ module: String,
    _ diagnostics: ActivationWeightedScaleSearchDiagnostics
  ) -> String {
    String(
      format: "  [%d/%d] %@ weighted=%+.4f%% raw=%+.4f%% changed=%d/%d",
      completed, total, module,
      diagnostics.calibrationWeightedMSEReductionPercent,
      diagnostics.rawMSEReductionPercent,
      diagnostics.changedGroupCount,
      diagnostics.groupCount
    )
  }

  private func fnv1a64(_ data: Data) -> String {
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    for byte in data {
      hash ^= UInt64(byte)
      hash &*= 0x0000_0100_0000_01b3
    }
    return String(format: "fnv1a64:%016llx", hash)
  }

  private func validateDirectory(_ url: URL, label: String) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw RescoreError.invalidInput("\(label) directory does not exist: \(url.path)")
    }
  }

  private func validateFile(_ url: URL, label: String) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
      !isDirectory.boolValue
    else {
      throw RescoreError.invalidInput("\(label) file does not exist: \(url.path)")
    }
  }
}

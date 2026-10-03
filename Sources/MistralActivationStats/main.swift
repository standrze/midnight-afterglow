import ArgumentParser
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MistralActivationScaleSearchCore
import WickQualitySupport
import WickModelSupport
import Tokenizers

private let statisticSuffix = ".input_second_moment"
private let statisticsFormat = "mistral_activation_stats_v1"

private struct CheckpointDescriptor: Sendable {
  var modelType: String
  var textModelType: String
  var hiddenLayerCount: Int
  var tiedWordEmbeddings: Bool
  var architectures: [String]
  var isQuantized: Bool
}

private struct CollectionPayload: Sendable {
  var samples: [ModelQualityCorpusSample]
  var segmentTokenLimit: Int
  var maximumTotalTokens: Int
  var expectedModulePaths: Set<String>
  var laguna: Bool
  var minimumExpertPositions: Int
  var layerwise: Bool
  var spoolDirectory: URL
}

private struct CollectedSample: Encodable, Sendable {
  var id: String
  var category: String?
  var originalTokenCount: Int
  var observedTokenCount: Int
  var segmentCount: Int
  var truncated: Bool
  var tokenIDFingerprint: String

  enum CodingKeys: String, CodingKey {
    case id, category, truncated
    case originalTokenCount = "original_token_count"
    case observedTokenCount = "observed_token_count"
    case segmentCount = "segment_count"
    case tokenIDFingerprint = "token_id_fingerprint"
  }
}

private struct CollectedModule: Sendable {
  var path: String
  var tensorKey: String
  var inputWidth: Int
  var observedPositionCount: Int
  var values: [Float]
  var minimum: Float
  var maximum: Float
  var mean: Double
}

private struct CollectedExpertModule: Sendable {
  var path: String
  var shape: [Int]
  var values: [Float]
  var counts: [Int]
  var minimumExpertPositions: Int
}

private struct ExpertModuleReport: Encodable, Sendable {
  var path: String
  var shape: [Int]
  var expert_position_counts: [Int]
  var minimum_expert_positions: Int
  var eligible_experts: [Bool]
}

private struct CollectedStatistics: Sendable {
  var device: String
  var sourceWeightDType: String
  var tokenIDFingerprint: String
  var observedTokenCount: Int
  var segmentCount: Int
  var modules: [CollectedModule]
  var samples: [CollectedSample]
  var expertModules: [CollectedExpertModule] = []
  var segmentTokenFingerprints: [String] = []
}

private struct ModuleReport: Encodable, Sendable {
  var path: String
  var tensorKey: String
  var inputWidth: Int
  var observedPositionCount: Int
  var minimumSecondMoment: Float
  var maximumSecondMoment: Float
  var meanSecondMoment: Double

  enum CodingKeys: String, CodingKey {
    case path
    case tensorKey = "tensor_key"
    case inputWidth = "input_width"
    case observedPositionCount = "observed_position_count"
    case minimumSecondMoment = "minimum_second_moment"
    case maximumSecondMoment = "maximum_second_moment"
    case meanSecondMoment = "mean_second_moment"
  }
}

private struct ActivationStatisticsReport: Encodable, Sendable {
  var format = 1
  var status = "measured"
  var algorithm = "input_channel_second_moment"
  var createdAt: String
  var sourceModel: String
  var modelType: String
  var textModelType: String
  var architectures: [String]
  var sourceWeightDType: String
  var sourceConfigFingerprint: String
  var sourceIndexFingerprint: String?
  var sourceWeightFingerprint: String? = nil
  var sourceWeightFingerprintMethod: String? = nil
  var corpusPath: String
  var corpusFingerprint: String
  var tokenIDFingerprint: String
  var outputPath: String
  var backend: String
  var device: String
  var addSpecialTokens = true
  var segmentation = "contiguous_nonoverlapping_independent"
  var segmentTokenLimit: Int
  var maximumTotalTokens: Int?
  var corpusSampleCount: Int
  var observedSampleCount: Int
  var observedSegmentCount: Int
  var observedTokenCount: Int
  var moduleCount: Int
  var mlxPeakMemoryBytes: Int
  var elapsedSeconds: Double
  var modules: [ModuleReport]
  var samples: [CollectedSample]
  var expertModules: [ExpertModuleReport] = []

  enum CodingKeys: String, CodingKey {
    case format, status, algorithm, architectures, backend, device, segmentation, modules, samples
    case expertModules = "expert_modules"
    case createdAt = "created_at"
    case sourceModel = "source_model"
    case modelType = "model_type"
    case textModelType = "text_model_type"
    case sourceWeightDType = "source_weight_dtype"
    case sourceConfigFingerprint = "source_config_fingerprint"
    case sourceIndexFingerprint = "source_index_fingerprint"
    case sourceWeightFingerprint = "source_weight_fingerprint"
    case sourceWeightFingerprintMethod = "source_weight_fingerprint_method"
    case corpusPath = "corpus_path"
    case corpusFingerprint = "corpus_fingerprint"
    case tokenIDFingerprint = "token_id_fingerprint"
    case outputPath = "output_path"
    case addSpecialTokens = "add_special_tokens"
    case segmentTokenLimit = "segment_token_limit"
    case maximumTotalTokens = "maximum_total_tokens"
    case corpusSampleCount = "corpus_sample_count"
    case observedSampleCount = "observed_sample_count"
    case observedSegmentCount = "observed_segment_count"
    case observedTokenCount = "observed_token_count"
    case moduleCount = "module_count"
    case mlxPeakMemoryBytes = "mlx_peak_memory_bytes"
    case elapsedSeconds = "elapsed_seconds"
  }
}

private enum ActivationStatisticsError: Error, LocalizedError {
  case invalidInput(String)
  case unsupportedCheckpoint(String)
  case invalidModelGraph(String)
  case collectionFailed(String)
  case invalidStatistics(String)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let detail):
      "Invalid activation-statistics input: \(detail)"
    case .unsupportedCheckpoint(let detail):
      "Unsupported activation-statistics checkpoint: \(detail)"
    case .invalidModelGraph(let detail):
      "Invalid model graph: \(detail)"
    case .collectionFailed(let detail):
      "Activation-statistics collection failed: \(detail)"
    case .invalidStatistics(let detail):
      "Invalid activation statistics: \(detail)"
    }
  }
}

private final class ActivationRecorder {
  private struct Entry {
    var inputWidth: Int
    var sumSquares: MLXArray
    var positionCount: Int
  }

  private let expectedWidths: [String: Int]
  private var entries = [String: Entry]()
  private var firstFailure: String?

  init(expectedWidths: [String: Int]) {
    self.expectedWidths = expectedWidths
  }

  func observe(path: String, input: MLXArray) {
    guard firstFailure == nil else { return }
    guard let expectedWidth = expectedWidths[path] else {
      firstFailure = "unexpected recording path '\(path)'"
      return
    }
    guard input.ndim >= 1, input.dim(-1) == expectedWidth else {
      firstFailure =
        "\(path) received shape \(input.shape); expected final dimension \(expectedWidth)"
      return
    }

    var positionCount = 1
    for dimension in input.shape.dropLast() {
      let (next, overflow) = positionCount.multipliedReportingOverflow(by: dimension)
      guard !overflow else {
        firstFailure = "position count overflow for \(path) with shape \(input.shape)"
        return
      }
      positionCount = next
    }
    guard positionCount > 0 else {
      firstFailure = "\(path) received an empty activation tensor with shape \(input.shape)"
      return
    }

    let reductionAxes = Array(0..<max(0, input.ndim - 1))
    let squared = MLX.square(input.asType(.float32))
    let batchSum =
      reductionAxes.isEmpty
      ? squared
      : squared.sum(axes: reductionAxes)

    if var entry = entries[path] {
      let (nextCount, overflow) = entry.positionCount.addingReportingOverflow(positionCount)
      guard !overflow else {
        firstFailure = "accumulated position count overflow for \(path)"
        return
      }
      entry.sumSquares = entry.sumSquares + batchSum
      entry.positionCount = nextCount
      entries[path] = entry
    } else {
      entries[path] = Entry(
        inputWidth: expectedWidth,
        sumSquares: batchSum,
        positionCount: positionCount
      )
    }
  }

  func evaluatePending() throws {
    try throwRecordedFailure()
    guard !entries.isEmpty else {
      throw ActivationStatisticsError.collectionFailed(
        "the model forward pass recorded no Linear inputs")
    }
    do {
      try MLX.checkedEval(entries.values.map(\.sumSquares))
    } catch {
      throw ActivationStatisticsError.collectionFailed(
        "MLX could not evaluate accumulated moments: \(error.localizedDescription)")
    }
    try throwRecordedFailure()
  }

  func finalize(expectedPaths: Set<String>) throws -> [CollectedModule] {
    try evaluatePending()
    let observedPaths = Set(entries.keys)
    let missing = expectedPaths.subtracting(observedPaths)
    let unexpected = observedPaths.subtracting(expectedPaths)
    guard missing.isEmpty, unexpected.isEmpty else {
      throw ActivationStatisticsError.invalidStatistics(
        "module coverage mismatch; missing [\(missing.sorted().joined(separator: ", "))], "
          + "unexpected [\(unexpected.sorted().joined(separator: ", "))]")
    }

    var moments = [String: MLXArray]()
    for path in expectedPaths.sorted() {
      guard let entry = entries[path], entry.positionCount > 0 else {
        throw ActivationStatisticsError.invalidStatistics(
          "\(path) has no observed activation positions")
      }
      moments[path] = (entry.sumSquares / Float(entry.positionCount)).asType(.float32)
    }
    do {
      try MLX.checkedEval(Array(moments.values))
    } catch {
      throw ActivationStatisticsError.invalidStatistics(
        "MLX could not materialize final moments: \(error.localizedDescription)")
    }

    var result = [CollectedModule]()
    result.reserveCapacity(expectedPaths.count)
    for path in expectedPaths.sorted() {
      guard let entry = entries[path], let moment = moments[path] else {
        throw ActivationStatisticsError.invalidStatistics(
          "internal result is missing \(path)")
      }
      let values = moment.asArray(Float.self)
      guard values.count == entry.inputWidth else {
        throw ActivationStatisticsError.invalidStatistics(
          "\(path) produced \(values.count) values; expected \(entry.inputWidth)")
      }
      guard values.allSatisfy({ $0.isFinite && $0 >= 0 }),
        values.contains(where: { $0 > 0 })
      else {
        throw ActivationStatisticsError.invalidStatistics(
          "\(path) contains non-finite, negative, or entirely zero second moments")
      }
      guard let minimum = values.min(), let maximum = values.max() else {
        throw ActivationStatisticsError.invalidStatistics("\(path) is empty")
      }
      let mean = values.reduce(0.0) { $0 + Double($1) } / Double(values.count)
      guard mean.isFinite, mean > 0 else {
        throw ActivationStatisticsError.invalidStatistics(
          "\(path) has an invalid mean second moment")
      }
      result.append(
        CollectedModule(
          path: path,
          tensorKey: path + statisticSuffix,
          inputWidth: entry.inputWidth,
          observedPositionCount: entry.positionCount,
          values: values,
          minimum: minimum,
          maximum: maximum,
          mean: mean
        ))
    }
    return result
  }

  private func throwRecordedFailure() throws {
    if let firstFailure {
      throw ActivationStatisticsError.collectionFailed(firstFailure)
    }
  }
}

private final class ActivationRecordingLinear: Linear {
  private let recordingPath: String
  private let recorder: ActivationRecorder

  init(path: String, linear: Linear, recorder: ActivationRecorder) {
    self.recordingPath = path
    self.recorder = recorder
    super.init(weight: linear.weight, bias: linear.bias)
    train(linear.training)
  }

  override func callAsFunction(_ x: MLXArray) -> MLXArray {
    recorder.observe(path: recordingPath, input: x)
    return super.callAsFunction(x)
  }
}

@main
private struct MistralActivationStats: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: CommandLine.arguments.first?.split(separator: "/").last.map(String.init)
      ?? "wick-mistral-activation-stats",
    abstract:
      "Collect BF16 Mistral dense or Laguna dense and expert-conditional activation moments for affine Q4 calibration."
  )

  @Argument(help: "Unquantized BF16 Mistral/Ministral or Laguna MLX checkpoint directory.")
  var model: String

  @Argument(help: "Deterministic plain-text JSONL calibration corpus.")
  var corpus: String

  @Argument(help: "New .safetensors statistics output path.")
  var output: String

  @Option(
    name: .customLong("segment-tokens"),
    help: "Maximum tokens per independent contiguous model segment (1...2048)."
  )
  var segmentTokens = 512

  @Option(
    name: .customLong("maximum-total-tokens"),
    help: "Deterministic corpus-prefix token cap; zero means all tokens."
  )
  var maximumTotalTokens = 0

  @Option(name: .customLong("minimum-expert-positions"),
    help: "Laguna experts below this selected-position count must retain template weights.")
  var minimumExpertPositions = 32

  @Flag(help: "Laguna only: stream one BF16 layer at a time through a temporary activation spool.")
  var layerwise = false

  @Option(name: .customLong("spool-directory"), help: "Parent directory for the private temporary Laguna activation spool.")
  var spoolDirectory: String?

  @Flag(help: "Run on CPU instead of the default MLX device.")
  var cpu = false

  @Flag(help: "Atomically replace existing statistics and report files.")
  var overwrite = false

  mutating func validate() throws {
    guard (1...2_048).contains(segmentTokens) else {
      throw ValidationError("--segment-tokens must be in 1...2048.")
    }
    guard minimumExpertPositions > 0 else {
      throw ValidationError("--minimum-expert-positions must be positive.")
    }
    guard maximumTotalTokens >= 0 else {
      throw ValidationError("--maximum-total-tokens must be nonnegative.")
    }
  }

  mutating func run() async throws {
    let modelURL = localURL(model, isDirectory: true)
    let corpusURL = localURL(corpus)
    let outputURL = localURL(output)
    let reportURL = outputURL.deletingPathExtension().appendingPathExtension("json")
    try validateInputs(
      modelURL: modelURL,
      corpusURL: corpusURL,
      outputURL: outputURL,
      reportURL: reportURL
    )

    let configURL = modelURL.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)
    let descriptor = try checkpointDescriptor(configData)
    try validateCheckpoint(descriptor, modelURL: modelURL)
    guard !layerwise || descriptor.modelType == "laguna" else {
      throw ValidationError("--layerwise is currently supported only for Laguna targets.")
    }
    let configFingerprint = fnv1a64Fingerprint(configData)
    let indexURL = modelURL.appendingPathComponent("model.safetensors.index.json")
    let indexFingerprint = try optionalFingerprint(indexURL)
    let sourceWeightFingerprint: String?
    if descriptor.modelType == "laguna" {
      print("Fingerprinting complete indexed source weights (bounded 8 MiB reads).")
      sourceWeightFingerprint = try IndexedSafetensorsFingerprint.compute(directory: modelURL)
    } else { sourceWeightFingerprint = nil }
    let samples = try ModelQualityCore.loadCorpus(from: corpusURL)
    let corpusFingerprint = ModelQualityCore.corpusFingerprint(samples)
    let expectedPaths = expectedProjectionPaths(descriptor)
    let payload = CollectionPayload(
      samples: samples,
      segmentTokenLimit: segmentTokens,
      maximumTotalTokens: maximumTotalTokens,
      expectedModulePaths: expectedPaths,
      laguna: descriptor.modelType == "laguna",
      minimumExpertPositions: minimumExpertPositions,
      layerwise: layerwise,
      spoolDirectory: spoolDirectory.map { localURL($0, isDirectory: true) }
        ?? FileManager.default.temporaryDirectory
    )

    let resourceLimits = try MLXResourceLimits.resolve(
      for: cpu ? .cpu : collectionEngine,
      physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory
    )
    let startedAt = ContinuousClock.now

    let collect: @Sendable () async throws -> CollectedStatistics = {
      Memory.peakMemory = 0
      if payload.layerwise {
        try MLXResourceGuard.apply(resourceLimits)
        return try await collectLagunaLayerwise(modelURL: modelURL, configData: configData, payload: payload)
      }
      if payload.laguna { await LagunaModelRegistration.register() }
      try MLXResourceGuard.apply(resourceLimits)
      let container = try await #huggingFaceLoadModelContainer(
        configuration: ModelConfiguration(directory: modelURL)
      )
      return try await container.perform(values: payload) { context, payload in
        context.model.train(false)
        let leaves = context.model.leafModules().flattened()
        let linearLeaves = leaves.compactMap { path, module -> (String, Linear)? in
          guard let linear = module as? Linear else { return nil }
          return (path, linear)
        }
        let actualPaths = Set(linearLeaves.map(\.0))
        guard !payload.laguna || context.model is LagunaModel else {
          throw ActivationStatisticsError.invalidModelGraph("expected the native Laguna teacher")
        }
        let expectedDensePaths = payload.laguna ? actualPaths : payload.expectedModulePaths
        let missing = expectedDensePaths.subtracting(actualPaths)
        let unexpected = actualPaths.subtracting(expectedDensePaths)
        guard missing.isEmpty, unexpected.isEmpty else {
          throw ActivationStatisticsError.invalidModelGraph(
            "Linear path mismatch; missing [\(missing.sorted().joined(separator: ", "))], "
              + "unexpected [\(unexpected.sorted().joined(separator: ", "))]")
        }

        var expectedWidths = [String: Int]()
        var weightDTypes = Set<DType>()
        var replacements = [(String, Module)]()
        replacements.reserveCapacity(linearLeaves.count)
        for (path, linear) in linearLeaves.sorted(by: { $0.0 < $1.0 }) {
          guard !(linear is Quantized) else {
            throw ActivationStatisticsError.invalidModelGraph(
              "\(path) is already quantized; collect statistics from the BF16 teacher")
          }
          guard linear.weight.ndim == 2, linear.weight.dim(-1) > 0 else {
            throw ActivationStatisticsError.invalidModelGraph(
              "\(path) has invalid weight shape \(linear.weight.shape)")
          }
          guard linear.weight.dtype == .bfloat16 else {
            throw ActivationStatisticsError.invalidModelGraph(
              "\(path) uses \(linear.weight.dtype), but this collector requires BF16 teacher weights"
            )
          }
          expectedWidths[path] = linear.weight.dim(-1)
          weightDTypes.insert(linear.weight.dtype)
        }
        guard weightDTypes == Set([DType.bfloat16]) else {
          throw ActivationStatisticsError.invalidModelGraph(
            "expected one BF16 source dtype, found \(weightDTypes)")
        }

        if payload.laguna {
          for (path, module) in leaves where module is SwitchLinear {
            guard let weight = Dictionary(uniqueKeysWithValues: module.parameters().flattened())["weight"],
              weight.ndim == 3, weight.dtype == .bfloat16
            else {
              throw ActivationStatisticsError.invalidModelGraph("\(path) requires stacked BF16 expert weights")
            }
          }
        }
        let recorder = ActivationRecorder(expectedWidths: expectedWidths)
        let expertRecorder = try payload.laguna
          ? LagunaRoutedActivationRecorder(minimumExpertPositions: payload.minimumExpertPositions) : nil
        if let laguna = context.model as? LagunaModel {
          try laguna.setRoutedActivationObserver(expertRecorder)
        }
        defer { try? (context.model as? LagunaModel)?.setRoutedActivationObserver(nil) }
        for (path, linear) in linearLeaves.sorted(by: { $0.0 < $1.0 }) {
          replacements.append(
            (path, ActivationRecordingLinear(path: path, linear: linear, recorder: recorder)))
        }
        try context.model.update(
          modules: ModuleChildren.unflattened(replacements),
          verify: [.noUnusedKeys]
        )
        context.model.train(false)

        var observedTokens = 0
        var observedSegments = 0
        var sampleReports = [CollectedSample]()
        var tokenSequences = [ModelQualityTokenSequence]()
        sampleReports.reserveCapacity(payload.samples.count)

        for (sampleIndex, sample) in payload.samples.enumerated() {
          let encoded = context.tokenizer.encode(text: sample.text, addSpecialTokens: true)
          guard !encoded.isEmpty else {
            throw ActivationStatisticsError.collectionFailed(
              "sample '\(sample.id)' encoded to no tokens")
          }
          let remaining =
            payload.maximumTotalTokens == 0
            ? encoded.count
            : max(0, payload.maximumTotalTokens - observedTokens)
          if remaining == 0 { break }
          let selected = Array(encoded.prefix(remaining))
          var sampleSegments = 0

          for offset in stride(from: 0, to: selected.count, by: payload.segmentTokenLimit) {
            let end = min(offset + payload.segmentTokenLimit, selected.count)
            let segment = Array(selected[offset..<end])
            let inputs = MLXArray(segment).reshaped(1, segment.count)
            LagunaRuntimeTuning.$useCompiledBlockTail.withValue(false) {
              _ = context.model(inputs, cache: nil)
            }
            try recorder.evaluatePending()
            try expertRecorder?.evaluatePending()
            Memory.clearCache()

            let segmentID = "\(sample.id)#\(sampleSegments)"
            tokenSequences.append(
              ModelQualityTokenSequence(sampleID: segmentID, tokenIDs: segment))
            sampleSegments += 1
            observedSegments += 1
            let (nextObservedTokens, overflow) = observedTokens.addingReportingOverflow(
              segment.count)
            guard !overflow else {
              throw ActivationStatisticsError.collectionFailed(
                "observed token count overflow")
            }
            observedTokens = nextObservedTokens
          }

          sampleReports.append(
            CollectedSample(
              id: sample.id,
              category: sample.category,
              originalTokenCount: encoded.count,
              observedTokenCount: selected.count,
              segmentCount: sampleSegments,
              truncated: selected.count < encoded.count,
              tokenIDFingerprint: ModelQualityCore.tokenIDFingerprint(selected)
            ))
          print(
            "sample \(sampleIndex + 1)/\(payload.samples.count) \(sample.id): "
              + "\(selected.count) tokens in \(sampleSegments) segment(s)"
          )
          if payload.maximumTotalTokens > 0,
            observedTokens >= payload.maximumTotalTokens
          {
            break
          }
        }

        guard observedTokens > 0, observedSegments > 0 else {
          throw ActivationStatisticsError.collectionFailed(
            "the configured corpus prefix contains no tokens")
        }
        let modules = try recorder.finalize(expectedPaths: expectedDensePaths)
        let expertModules = try expertRecorder?.finalize().map { result in
          CollectedExpertModule(
            path: result.path, shape: result.secondMoments.shape,
            values: result.secondMoments.asArray(Float.self),
            counts: result.expertPositionCounts,
            minimumExpertPositions: result.minimumExpertPositions)
        } ?? []
        if payload.laguna {
          let expectedExpertPaths = Set(leaves.compactMap { path, module in
            module is SwitchLinear ? path : nil
          })
          guard Set(expertModules.map(\.path)) == expectedExpertPaths else {
            throw ActivationStatisticsError.invalidStatistics("Laguna routed projection coverage mismatch")
          }
          for module in expertModules {
            guard module.counts.reduce(0, +) > 0 else {
              throw ActivationStatisticsError.invalidStatistics("no routed positions for \(module.path)")
            }
          }
        }
        let moduleCounts = Set(modules.map(\.observedPositionCount))
        guard moduleCounts == Set([observedTokens]) else {
          throw ActivationStatisticsError.invalidStatistics(
            "dense modules observed inconsistent position counts \(moduleCounts.sorted()); "
              + "expected \(observedTokens)")
        }

        return CollectedStatistics(
          device: Device.defaultDevice().deviceType?.rawValue ?? "unknown",
          sourceWeightDType: "bfloat16",
          tokenIDFingerprint: ModelQualityCore.combinedTokenIDFingerprint(tokenSequences),
          observedTokenCount: observedTokens,
          segmentCount: observedSegments,
          modules: modules,
          samples: sampleReports,
          expertModules: expertModules,
          segmentTokenFingerprints: tokenSequences.map { ModelQualityCore.tokenIDFingerprint($0.tokenIDs) }
        )
      }
    }

    let collected: CollectedStatistics
    if cpu {
      collected = try await Device.withDefaultDevice(.cpu, collect)
    } else {
      collected = try await collect()
    }
    let elapsedSeconds = seconds(startedAt.duration(to: .now))
    let peakMemory = Memory.peakMemory

    var arrays = [String: MLXArray]()
    arrays.reserveCapacity(collected.modules.count)
    for module in collected.modules {
      arrays[module.tensorKey] = MLXArray(module.values)
    }
    try MLX.checkedEval(Array(arrays.values))

    for module in collected.expertModules {
      arrays[module.path + statisticSuffix] = MLXArray(module.values).reshaped(module.shape)
      arrays[module.path + ".expert_position_count"] = MLXArray(module.counts.map(Int32.init))
    }
    let lagunaStatistics = descriptor.modelType == "laguna"
    var metadata = [
      "format": lagunaStatistics ? "laguna_expert_activation_stats_v1" : statisticsFormat,
      "algorithm": lagunaStatistics ? "expert_conditional_input_channel_second_moment" : "input_channel_second_moment",
      "model_type": descriptor.modelType,
      "source_model": modelURL.path,
      "source_config_fingerprint": configFingerprint,
      "corpus_fingerprint": corpusFingerprint,
      "token_id_fingerprint": collected.tokenIDFingerprint,
      "observed_token_count": String(collected.observedTokenCount),
      "module_count": String(collected.modules.count),
      "segment_token_limit": String(segmentTokens),
      "add_special_tokens": "true",
      "dtype": "float32",
    ]
    if lagunaStatistics {
      metadata["collection_strategy"] = layerwise ? "layer_major_activation_spool" : "full_model"
      metadata["source_weight_fingerprint_method"] = IndexedSafetensorsFingerprint.method
      metadata["source_weight_fingerprint"] = sourceWeightFingerprint!
      metadata["expert_module_count"] = String(collected.expertModules.count)
      metadata["minimum_expert_positions"] = String(minimumExpertPositions)
      metadata["insufficient_coverage_policy"] = "retain_template_expert"
      metadata["router_weighting"] = "conditional_on_selection_no_routing_score_weight"
      metadata["sample_token_fingerprints"] = String(decoding:
        try JSONEncoder().encode(Set(collected.samples.map(\.tokenIDFingerprint)).sorted()), as: UTF8.self)
      metadata["segment_token_fingerprints"] = String(decoding:
        try JSONEncoder().encode(Set(collected.segmentTokenFingerprints).sorted()), as: UTF8.self)
    }
    if let indexFingerprint {
      metadata["source_index_fingerprint"] = indexFingerprint
    }
    try writeSafetensorsAtomically(arrays, metadata: metadata, to: outputURL)

    let moduleReports = collected.modules.map {
      ModuleReport(
        path: $0.path,
        tensorKey: $0.tensorKey,
        inputWidth: $0.inputWidth,
        observedPositionCount: $0.observedPositionCount,
        minimumSecondMoment: $0.minimum,
        maximumSecondMoment: $0.maximum,
        meanSecondMoment: $0.mean
      )
    }
    var report = ActivationStatisticsReport(
      createdAt: ISO8601DateFormatter().string(from: Date()),
      sourceModel: modelURL.path,
      modelType: descriptor.modelType,
      textModelType: descriptor.textModelType,
      architectures: descriptor.architectures,
      sourceWeightDType: collected.sourceWeightDType,
      sourceConfigFingerprint: configFingerprint,
      sourceIndexFingerprint: indexFingerprint,
      corpusPath: corpusURL.path,
      corpusFingerprint: corpusFingerprint,
      tokenIDFingerprint: collected.tokenIDFingerprint,
      outputPath: outputURL.path,
      backend: backendName,
      device: collected.device,
      segmentTokenLimit: segmentTokens,
      maximumTotalTokens: maximumTotalTokens == 0 ? nil : maximumTotalTokens,
      corpusSampleCount: samples.count,
      observedSampleCount: collected.samples.count,
      observedSegmentCount: collected.segmentCount,
      observedTokenCount: collected.observedTokenCount,
      moduleCount: collected.modules.count,
      mlxPeakMemoryBytes: peakMemory,
      elapsedSeconds: elapsedSeconds,
      modules: moduleReports,
      samples: collected.samples
    )
    if lagunaStatistics {
      report.algorithm = "expert_conditional_input_channel_second_moment"
      report.sourceWeightFingerprint = sourceWeightFingerprint
      report.sourceWeightFingerprintMethod = IndexedSafetensorsFingerprint.method
      report.expertModules = collected.expertModules.map { module in
        ExpertModuleReport(path: module.path, shape: module.shape,
          expert_position_counts: module.counts, minimum_expert_positions: module.minimumExpertPositions,
          eligible_experts: module.counts.map { $0 >= module.minimumExpertPositions })
      }
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: reportURL, options: .atomic)

    print(
      "Collected \(collected.modules.count) module statistics across "
        + "\(collected.observedTokenCount) tokens in \(collected.segmentCount) segment(s)."
    )
    print("Wrote \(outputURL.path)")
    print("Wrote \(reportURL.path)")
  }

  private func validateInputs(
    modelURL: URL,
    corpusURL: URL,
    outputURL: URL,
    reportURL: URL
  ) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: modelURL.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw ActivationStatisticsError.invalidInput(
        "model directory does not exist: \(modelURL.path)")
    }
    isDirectory = false
    guard FileManager.default.fileExists(atPath: corpusURL.path, isDirectory: &isDirectory),
      !isDirectory.boolValue
    else {
      throw ActivationStatisticsError.invalidInput(
        "corpus file does not exist: \(corpusURL.path)")
    }
    guard outputURL.pathExtension == "safetensors" else {
      throw ActivationStatisticsError.invalidInput(
        "output must use the .safetensors extension")
    }
    guard outputURL != corpusURL, reportURL != corpusURL else {
      throw ActivationStatisticsError.invalidInput(
        "output and report paths must not replace the corpus")
    }
    let modelPrefix = modelURL.path.hasSuffix("/") ? modelURL.path : modelURL.path + "/"
    guard !outputURL.path.hasPrefix(modelPrefix), !reportURL.path.hasPrefix(modelPrefix) else {
      throw ActivationStatisticsError.invalidInput(
        "statistics must not be written inside the model directory")
    }
    if !overwrite {
      for url in [outputURL, reportURL]
      where FileManager.default.fileExists(atPath: url.path) {
        throw ActivationStatisticsError.invalidInput(
          "output already exists (pass --overwrite to replace it): \(url.path)")
      }
    }
  }
}

private func checkpointDescriptor(_ data: Data) throws -> CheckpointDescriptor {
  guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
    let modelType = json["model_type"] as? String
  else {
    throw ActivationStatisticsError.invalidInput(
      "config.json has no string model_type")
  }
  let text = (json["text_config"] as? [String: Any]) ?? json
  guard let hiddenLayerCount = (text["num_hidden_layers"] as? NSNumber)?.intValue,
    hiddenLayerCount > 0
  else {
    throw ActivationStatisticsError.invalidInput(
      "config.json has no positive num_hidden_layers")
  }
  let textModelType = (text["model_type"] as? String) ?? modelType
  let tiedWordEmbeddings =
    (text["tie_word_embeddings"] as? Bool)
    ?? (json["tie_word_embeddings"] as? Bool)
    ?? false
  let architectures = json["architectures"] as? [String] ?? []
  let isQuantized =
    json["quantization"].map { !($0 is NSNull) } ?? false
    || (json["quantization_config"].map { !($0 is NSNull) } ?? false)
  return CheckpointDescriptor(
    modelType: modelType,
    textModelType: textModelType,
    hiddenLayerCount: hiddenLayerCount,
    tiedWordEmbeddings: tiedWordEmbeddings,
    architectures: architectures,
    isQuantized: isQuantized
  )
}

private func validateCheckpoint(
  _ descriptor: CheckpointDescriptor,
  modelURL: URL
) throws {
  let supportedTypes = Set(["mistral3", "ministral3", "laguna"])
  guard supportedTypes.contains(descriptor.modelType),
    supportedTypes.contains(descriptor.textModelType)
  else {
    throw ActivationStatisticsError.unsupportedCheckpoint(
      "expected Mistral3/Ministral3/Laguna, found top-level '\(descriptor.modelType)' and text '\(descriptor.textModelType)'"
    )
  }
  guard !descriptor.architectures.contains(where: { $0.lowercased().contains("dflash") }) else {
    throw ActivationStatisticsError.unsupportedCheckpoint("DFlash is not the Laguna target teacher")
  }
  guard !descriptor.isQuantized else {
    throw ActivationStatisticsError.unsupportedCheckpoint(
      "\(modelURL.path) declares quantization; use its unquantized BF16 teacher")
  }
}

private func expectedProjectionPaths(_ descriptor: CheckpointDescriptor) -> Set<String> {
  var result = Set<String>()
  for layer in 0..<descriptor.hiddenLayerCount {
    let prefix = "model.layers.\(layer)"
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
      result.insert("\(prefix).self_attn.\(projection)")
    }
    for projection in ["gate_proj", "up_proj", "down_proj"] {
      result.insert("\(prefix).mlp.\(projection)")
    }
  }
  if !descriptor.tiedWordEmbeddings {
    result.insert("lm_head")
  }
  return result
}

private func writeSafetensorsAtomically(
  _ arrays: [String: MLXArray],
  metadata: [String: String],
  to outputURL: URL
) throws {
  let fileManager = FileManager.default
  let parent = outputURL.deletingLastPathComponent()
  try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
  let temporaryURL = parent.appendingPathComponent(
    ".\(outputURL.lastPathComponent).\(UUID().uuidString).partial.safetensors")
  defer { try? fileManager.removeItem(at: temporaryURL) }
  try MLX.save(arrays: arrays, metadata: metadata, url: temporaryURL)
  if fileManager.fileExists(atPath: outputURL.path) {
    _ = try fileManager.replaceItemAt(outputURL, withItemAt: temporaryURL)
  } else {
    try fileManager.moveItem(at: temporaryURL, to: outputURL)
  }
}

private func optionalFingerprint(_ url: URL) throws -> String? {
  guard FileManager.default.fileExists(atPath: url.path) else { return nil }
  return fnv1a64Fingerprint(try Data(contentsOf: url))
}

private func fnv1a64Fingerprint(_ data: Data) -> String {
  var value: UInt64 = 0xcbf2_9ce4_8422_2325
  for byte in data {
    value ^= UInt64(byte)
    value &*= 0x100_0000_01b3
  }
  let hex = String(value, radix: 16)
  return "fnv1a64:" + String(repeating: "0", count: 16 - hex.count) + hex
}

private func localURL(_ path: String, isDirectory: Bool = false) -> URL {
  let expanded = NSString(string: path).expandingTildeInPath
  return URL(fileURLWithPath: expanded, isDirectory: isDirectory).standardizedFileURL
}

private func seconds(_ duration: Duration) -> Double {
  let components = duration.components
  return Double(components.seconds)
    + Double(components.attoseconds) / 1_000_000_000_000_000_000
}

private var backendName: String {
  #if MLX_METAL_BACKEND
    "metal"
  #elseif MLX_CUDA_BACKEND
    "cuda"
  #elseif MLX_CPU_BACKEND
    "cpu"
  #else
    "unknown"
  #endif
}

private var collectionEngine: ModelEngine {
  #if MLX_METAL_BACKEND
    .metal
  #elseif MLX_CUDA_BACKEND
    .cuda
  #else
    .cpu
  #endif
}


/// Release Foundation file/serialization temporaries on each bounded unit of
/// work. MLX arrays and returned Swift values retain their own storage.
private func withCalibrationAutoreleasePool<Result>(
  _ body: () throws -> Result
) rethrows -> Result {
  #if canImport(ObjectiveC)
  return try autoreleasepool(invoking: body)
  #else
  return try body()
  #endif
}

/// Layer-major traversal preserves each independent segment's exact tokenizer
/// input and model operations while retaining only one BF16 block at a time.
private func collectLagunaLayerwise(
  modelURL: URL, configData: Data, payload: CollectionPayload
) async throws -> CollectedStatistics {
  let configuration = try JSONDecoder().decode(LagunaConfiguration.self, from: configData)
  let tokenizer = try await #huggingFaceTokenizerLoader().load(from: modelURL)
  let reader = try SelectiveSafetensorsReader(directory: modelURL)
  let keys = reader.keys
  func sourceKey(_ normalized: String) throws -> String {
    if keys.contains(normalized) { return normalized }
    let wrapped = "language_model." + normalized
    guard keys.contains(wrapped) else {
      throw ActivationStatisticsError.invalidInput("missing source tensor \(normalized)")
    }
    return wrapped
  }
  var sequences = [ModelQualityTokenSequence]()
  var samples = [CollectedSample]()
  var observedTokens = 0
  for sample in payload.samples {
    let encoded = tokenizer.encode(text: sample.text, addSpecialTokens: true)
    guard !encoded.isEmpty else {
      throw ActivationStatisticsError.collectionFailed("sample '\(sample.id)' encoded to no tokens")
    }
    let remaining = payload.maximumTotalTokens == 0 ? encoded.count
      : max(0, payload.maximumTotalTokens - observedTokens)
    if remaining == 0 { break }
    let selected = Array(encoded.prefix(remaining))
    var segments = 0
    for offset in stride(from: 0, to: selected.count, by: payload.segmentTokenLimit) {
      let end = min(offset + payload.segmentTokenLimit, selected.count)
      sequences.append(ModelQualityTokenSequence(sampleID: "\(sample.id)#\(segments)",
        tokenIDs: Array(selected[offset..<end])))
      segments += 1
    }
    observedTokens += selected.count
    samples.append(CollectedSample(id: sample.id, category: sample.category,
      originalTokenCount: encoded.count, observedTokenCount: selected.count,
      segmentCount: segments, truncated: selected.count < encoded.count,
      tokenIDFingerprint: ModelQualityCore.tokenIDFingerprint(selected)))
  }
  guard observedTokens > 0 else { throw ActivationStatisticsError.invalidInput("empty calibration corpus prefix") }
  let spoolParent = payload.spoolDirectory.standardizedFileURL.resolvingSymlinksInPath()
  let sourceDirectory = modelURL.standardizedFileURL.resolvingSymlinksInPath()
  guard spoolParent != sourceDirectory, !spoolParent.path.hasPrefix(sourceDirectory.path + "/") else {
    throw ActivationStatisticsError.invalidInput("activation spool must be outside the source checkpoint")
  }
  try FileManager.default.createDirectory(at: spoolParent, withIntermediateDirectories: true)
  let spool = spoolParent.appendingPathComponent("laguna-activations-\(UUID().uuidString)")
  try FileManager.default.createDirectory(at: spool, withIntermediateDirectories: false)
  defer { try? FileManager.default.removeItem(at: spool) }
  let (hiddenElements, firstOverflow) = observedTokens.multipliedReportingOverflow(by: configuration.calibrationHiddenSize)
  let (spoolBytes, secondOverflow) = hiddenElements.multipliedReportingOverflow(by: 4)
  guard !firstOverflow, !secondOverflow else {
    throw ActivationStatisticsError.invalidInput("activation spool size overflow")
  }
  if let available = try? spoolParent.resourceValues(forKeys: [.volumeAvailableCapacityKey]).volumeAvailableCapacity,
    available < spoolBytes + 1_048_576
  { throw ActivationStatisticsError.invalidInput("insufficient disk space for two BF16 activation stages") }
  print("Layerwise Laguna: \(sequences.count) independent segments; two-stage BF16 payload <= \(spoolBytes) bytes")
  func segmentURL(_ stage: URL, _ index: Int) -> URL {
    stage.appendingPathComponent(String(format: "segment-%06d.safetensors", index))
  }
  var currentStage = spool.appendingPathComponent("stage-0")
  try FileManager.default.createDirectory(at: currentStage, withIntermediateDirectories: false)
  func embedSegments() throws {
    let embedding = try reader.read(sourceKey("model.embed_tokens.weight"))
    guard embedding.dtype == .bfloat16, embedding.ndim == 2,
      embedding.dim(-1) == configuration.calibrationHiddenSize
    else { throw ActivationStatisticsError.invalidModelGraph("expected BF16 Laguna embedding") }
    for (index, segment) in sequences.enumerated() {
      try withCalibrationAutoreleasePool {
        let tokens = MLXArray(segment.tokenIDs).reshaped(1, segment.tokenIDs.count)
        let hidden = embedding[tokens]
        try MLX.checkedEval(hidden)
        try MLX.save(arrays: ["hidden": hidden], url: segmentURL(currentStage, index))
      }
    }
  }
  try withCalibrationAutoreleasePool { try embedSegments() }
  Memory.clearCache()
  var allModules = [CollectedModule]()
  var allExperts = [CollectedExpertModule]()
  for layer in 0..<configuration.calibrationLayerCount {
    let nextStage = spool.appendingPathComponent("stage-\(layer + 1)")
    try FileManager.default.createDirectory(at: nextStage, withIntermediateDirectories: false)
    func processLayer() throws -> ([CollectedModule], [CollectedExpertModule]) {
      let prefix = "model.layers.\(layer)."
      let layerKeys = keys.filter { $0.hasPrefix(prefix) || $0.hasPrefix("language_model." + prefix) }
      guard !layerKeys.isEmpty else { throw ActivationStatisticsError.invalidInput("missing source layer \(layer)") }
      let block = try LagunaCalibrationBlock(configuration: configuration, layerIndex: layer,
        sourceWeights: reader.read(keys: layerKeys))
      let linears = block.module.leafModules().flattened().compactMap { path, module -> (String, Linear)? in
        guard let linear = module as? Linear else { return nil }
        return (path, linear)
      }
      var widths = [String: Int]()
      for (path, linear) in linears {
        guard linear.weight.dtype == .bfloat16, linear.weight.ndim == 2 else {
          throw ActivationStatisticsError.invalidModelGraph("expected BF16 Linear in layer \(layer): \(path)")
        }
        widths[block.modulePrefix + "." + path] = linear.weight.dim(-1)
      }
      let dense = ActivationRecorder(expectedWidths: widths)
      let experts = try LagunaRoutedActivationRecorder(minimumExpertPositions: payload.minimumExpertPositions)
      let hasExperts = block.module.leafModules().flattened().contains { $0.1 is SwitchLinear }
      if hasExperts { try block.setRoutedActivationObserver(experts) }
      let replacements: [(String, Module)] = linears.map { path, linear in
        (path, ActivationRecordingLinear(path: block.modulePrefix + "." + path, linear: linear, recorder: dense))
      }
      try block.module.update(modules: ModuleChildren.unflattened(replacements), verify: [.noUnusedKeys])
      for (index, segment) in sequences.enumerated() {
        try withCalibrationAutoreleasePool {
          let arrays = try MLX.loadArrays(url: segmentURL(currentStage, index), stream: .cpu)
          guard let hidden = arrays["hidden"], hidden.shape == [1, segment.tokenIDs.count, configuration.calibrationHiddenSize],
            hidden.dtype == .bfloat16
          else { throw ActivationStatisticsError.invalidStatistics("activation spool shape/dtype mismatch") }
          let next = block(hidden)
          try MLX.checkedEval(next)
          try dense.evaluatePending()
          if hasExperts { try experts.evaluatePending() }
          try MLX.save(arrays: ["hidden": next], url: segmentURL(nextStage, index))
        }
        Memory.clearCache()
      }
      let denseResults = try dense.finalize(expectedPaths: Set(widths.keys))
      guard denseResults.allSatisfy({ $0.observedPositionCount == observedTokens }) else {
        throw ActivationStatisticsError.invalidStatistics("layerwise dense position coverage mismatch")
      }
      let expertResults = try hasExperts ? experts.finalize().map {
        CollectedExpertModule(path: $0.path, shape: $0.secondMoments.shape,
          values: $0.secondMoments.asArray(Float.self), counts: $0.expertPositionCounts,
          minimumExpertPositions: $0.minimumExpertPositions)
      } : []
      return (denseResults, expertResults)
    }
    let (dense, experts) = try withCalibrationAutoreleasePool { try processLayer() }
    allModules.append(contentsOf: dense)
    allExperts.append(contentsOf: experts)
    try FileManager.default.removeItem(at: currentStage)
    currentStage = nextStage
    Memory.clearCache()
    print("Layerwise Laguna: completed layer \(layer + 1)/\(configuration.calibrationLayerCount)")
  }
  if !configuration.calibrationTiesWordEmbeddings {
    let headDescription = try reader.description(for: sourceKey("lm_head.weight"))
    guard headDescription.dtype == .bfloat16, headDescription.shape.last == configuration.calibrationHiddenSize else {
      throw ActivationStatisticsError.invalidModelGraph("invalid BF16 LM head")
    }
    let norm = try reader.read(sourceKey("model.norm.weight"))
    let path = "language_model.lm_head"
    let recorder = ActivationRecorder(expectedWidths: [path: configuration.calibrationHiddenSize])
    for index in sequences.indices {
      try withCalibrationAutoreleasePool {
        let arrays = try MLX.loadArrays(url: segmentURL(currentStage, index), stream: .cpu)
        guard let hidden = arrays["hidden"] else { throw ActivationStatisticsError.invalidStatistics("missing final hidden state") }
        let normalized = MLXFast.rmsNorm(hidden, weight: norm, eps: configuration.calibrationRMSNormEpsilon)
        recorder.observe(path: path, input: normalized)
        try recorder.evaluatePending()
      }
      Memory.clearCache()
    }
    allModules.append(contentsOf: try recorder.finalize(expectedPaths: [path]))
  }
  return CollectedStatistics(device: Device.defaultDevice().deviceType?.rawValue ?? "unknown",
    sourceWeightDType: "bfloat16",
    tokenIDFingerprint: ModelQualityCore.combinedTokenIDFingerprint(sequences),
    observedTokenCount: observedTokens, segmentCount: sequences.count,
    modules: allModules.sorted { $0.path < $1.path }, samples: samples,
    expertModules: allExperts.sorted { $0.path < $1.path },
    segmentTokenFingerprints: sequences.map { ModelQualityCore.tokenIDFingerprint($0.tokenIDs) })
}

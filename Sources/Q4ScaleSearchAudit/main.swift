import ArgumentParser
import Foundation
import MLX
import MLXLMCommon

private struct SafetensorsIndex: Decodable {
  var weightMap: [String: String]

  enum CodingKeys: String, CodingKey {
    case weightMap = "weight_map"
  }
}

private struct TensorAuditResult: Codable {
  var tensor: String
  var shard: String
  var shape: [Int]
  var dtype: String
  var elements: Int64
  var groups: Int64
  var changedGroups: Int64
  var changedGroupFraction: Double
  var standardMSE: Double
  var searchedMSE: Double
  var standardRelativeMSE: Double
  var searchedRelativeMSE: Double
  var mseReductionPercent: Double
  var standardStoredBytes: Int64
  var searchedStoredBytes: Int64
  var standardConversionMS: Double
  var searchedConversionMS: Double

  enum CodingKeys: String, CodingKey {
    case tensor, shard, shape, dtype, elements, groups
    case changedGroups = "changed_groups"
    case changedGroupFraction = "changed_group_fraction"
    case standardMSE = "standard_mse"
    case searchedMSE = "searched_mse"
    case standardRelativeMSE = "standard_relative_mse"
    case searchedRelativeMSE = "searched_relative_mse"
    case mseReductionPercent = "mse_reduction_percent"
    case standardStoredBytes = "standard_stored_bytes"
    case searchedStoredBytes = "searched_stored_bytes"
    case standardConversionMS = "standard_conversion_ms"
    case searchedConversionMS = "searched_conversion_ms"
  }
}

private struct AuditTotals: Codable {
  var tensors: Int
  var elements: Int64
  var groups: Int64
  var changedGroups: Int64
  var changedGroupFraction: Double
  var standardMSE: Double
  var searchedMSE: Double
  var mseReductionPercent: Double
  var tensorsImproved: Int
  var passesFifteenPercentMSEGate: Bool

  enum CodingKeys: String, CodingKey {
    case tensors, elements, groups
    case changedGroups = "changed_groups"
    case changedGroupFraction = "changed_group_fraction"
    case standardMSE = "standard_mse"
    case searchedMSE = "searched_mse"
    case mseReductionPercent = "mse_reduction_percent"
    case tensorsImproved = "tensors_improved"
    case passesFifteenPercentMSEGate = "passes_fifteen_percent_mse_gate"
  }
}

private struct AuditReport: Codable {
  var format: Int = 2
  var status: String = "experimental_measured_candidate"
  var algorithm: String = "q4r8_affine_scale_search_ls2"
  var createdAt: String
  var sourceModel: String
  var device: String
  var bits: Int = 4
  var groupSize: Int = 64
  var searchFactors: [Double] = [
    0.75, 0.8125, 0.875, 0.9375, 1.0,
    1.0625, 1.125, 1.1875, 1.25,
  ]
  var biasRefinementIterations: Int = 1
  var jointAffineRefinementIterations: Int = 2
  var tensors: [TensorAuditResult]
  var totals: AuditTotals

  enum CodingKeys: String, CodingKey {
    case format, status, algorithm, device, bits, tensors, totals
    case createdAt = "created_at"
    case sourceModel = "source_model"
    case groupSize = "group_size"
    case searchFactors = "search_factors"
    case biasRefinementIterations = "bias_refinement_iterations"
    case jointAffineRefinementIterations = "joint_affine_refinement_iterations"
  }
}

private enum AuditError: Error, LocalizedError {
  case invalidInput(String)
  case invalidTensor(String)
  case invariant(String)

  var errorDescription: String? {
    switch self {
    case .invalidInput(let message): "Invalid audit input: \(message)"
    case .invalidTensor(let message): "Invalid audit tensor: \(message)"
    case .invariant(let message): "Q4 scale-search invariant failed: \(message)"
    }
  }
}

@main
private struct Q4ScaleSearchAudit: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: CommandLine.arguments.first?.split(separator: "/").last.map(String.init)
      ?? "wick-q4-scale-search-audit",
    abstract: "Measure standard and searched affine Q4 grids on real safetensors weights."
  )

  @Argument(help: "Unquantized model directory containing model.safetensors.index.json.")
  var modelDirectory: String

  @Argument(help: "New JSON audit report to write.")
  var output: String

  @Option(
    name: .customLong("tensor"),
    help: "Exact source tensor key to audit; repeat to override the representative Laguna sample."
  )
  var requestedTensors: [String] = []

  @Option(help: "Limit the selected tensor count after deterministic sorting; zero means no limit.")
  var limit = 0

  @Flag(help: "Run quantization operations on CPU instead of the default GPU.")
  var cpu = false

  @Flag(help: "Replace an existing report.")
  var overwrite = false

  mutating func validate() throws {
    guard limit >= 0 else {
      throw ValidationError("--limit must be zero or positive.")
    }
    let normalized = requestedTensors.map {
      $0.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    guard normalized.allSatisfy({ !$0.isEmpty }), Set(normalized).count == normalized.count else {
      throw ValidationError("--tensor values must be non-empty and unique.")
    }
  }

  mutating func run() throws {
    #if os(Linux)
      defer { clearStreams() }
    #endif

    if cpu {
      try Device.withDefaultDevice(.cpu) {
        try execute()
      }
    } else {
      try execute()
    }
  }

  private func execute() throws {
    let sourceURL = URL(fileURLWithPath: modelDirectory).standardizedFileURL
    let outputURL = URL(fileURLWithPath: output).standardizedFileURL
    guard sourceURL != outputURL else {
      throw AuditError.invalidInput("source and report paths must differ")
    }
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: sourceURL.path, isDirectory: &isDirectory),
      isDirectory.boolValue
    else {
      throw AuditError.invalidInput("model directory does not exist: \(sourceURL.path)")
    }
    if FileManager.default.fileExists(atPath: outputURL.path), !overwrite {
      throw AuditError.invalidInput("report exists; choose a new path or pass --overwrite")
    }

    let indexURL = sourceURL.appendingPathComponent("model.safetensors.index.json")
    let index = try JSONDecoder().decode(
      SafetensorsIndex.self,
      from: Data(contentsOf: indexURL)
    )
    let selected = try selectedTensorNames(from: index)
    print(
      "AffineScaleSearch-Q4R8 real-weight audit: \(selected.count) tensor(s) on "
        + "\(String(describing: Device.defaultDevice()))"
    )

    var results: [TensorAuditResult] = []
    results.reserveCapacity(selected.count)
    for (offset, tensor) in selected.enumerated() {
      guard let shard = index.weightMap[tensor] else {
        throw AuditError.invalidInput("index does not map tensor \(tensor)")
      }
      let result = try autoreleasepool {
        try auditTensor(
          tensor,
          shard: shard,
          sourceURL: sourceURL
        )
      }
      results.append(result)
      Memory.clearCache()
      print(
        "[\(offset + 1)/\(selected.count)] \(tensor) shape=\(result.shape) "
          + "mse_reduction=\(formatted(result.mseReductionPercent))% "
          + "changed_groups=\(formatted(result.changedGroupFraction * 100))%"
      )
    }

    let totals = try aggregate(results)
    let report = AuditReport(
      createdAt: ISO8601DateFormatter().string(from: Date()),
      sourceModel: sourceURL.path,
      device: String(describing: Device.defaultDevice()),
      tensors: results,
      totals: totals
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    try encoder.encode(report).write(to: outputURL, options: .atomic)

    print(
      "aggregate: standard_mse=\(totals.standardMSE) searched_mse=\(totals.searchedMSE) "
        + "reduction=\(formatted(totals.mseReductionPercent))% "
        + "improved=\(totals.tensorsImproved)/\(totals.tensors) "
        + "15%_gate=\(totals.passesFifteenPercentMSEGate)"
    )
    print("Wrote \(outputURL.path)")
  }

  private func selectedTensorNames(from index: SafetensorsIndex) throws -> [String] {
    let available = Set(index.weightMap.keys)
    var selected: [String]
    if requestedTensors.isEmpty {
      selected = representativeLagunaTensors().filter(available.contains)
      guard !selected.isEmpty else {
        throw AuditError.invalidInput(
          "none of the representative Laguna tensor names exist; pass one or more --tensor values"
        )
      }
    } else {
      selected = requestedTensors.map {
        $0.trimmingCharacters(in: .whitespacesAndNewlines)
      }
      let missing = selected.filter { !available.contains($0) }
      guard missing.isEmpty else {
        throw AuditError.invalidInput("unknown tensor(s): \(missing.joined(separator: ", "))")
      }
    }
    selected.sort()
    if limit > 0 {
      selected = Array(selected.prefix(limit))
    }
    return selected
  }

  private func representativeLagunaTensors() -> [String] {
    var names = ["lm_head.weight"]
    for layer in [0, 1, 20, 39] {
      let prefix = "model.layers.\(layer)"
      names.append("\(prefix).self_attn.q_proj.weight")
      names.append("\(prefix).self_attn.o_proj.weight")
      if layer == 0 {
        names.append("\(prefix).mlp.gate_proj.weight")
        names.append("\(prefix).mlp.up_proj.weight")
        names.append("\(prefix).mlp.down_proj.weight")
      } else {
        names.append("\(prefix).mlp.shared_expert.gate_proj.weight")
        names.append("\(prefix).mlp.shared_expert.up_proj.weight")
        names.append("\(prefix).mlp.shared_expert.down_proj.weight")
        for expert in [0, 127, 255] {
          names.append("\(prefix).mlp.experts.\(expert).gate_proj.weight")
          names.append("\(prefix).mlp.experts.\(expert).up_proj.weight")
          names.append("\(prefix).mlp.experts.\(expert).down_proj.weight")
        }
      }
    }
    return names
  }

  private func auditTensor(
    _ tensor: String,
    shard: String,
    sourceURL: URL
  ) throws -> TensorAuditResult {
    let shardURL = sourceURL.appendingPathComponent(shard)
    guard FileManager.default.fileExists(atPath: shardURL.path) else {
      throw AuditError.invalidInput("missing shard \(shard)")
    }
    let arrays = try loadArrays(url: shardURL, stream: .cpu)
    guard let source = arrays[tensor] else {
      throw AuditError.invalidInput("shard \(shard) does not contain \(tensor)")
    }
    guard source.ndim >= 2, let inputWidth = source.shape.last,
      inputWidth > 0, inputWidth % 64 == 0
    else {
      throw AuditError.invalidTensor(
        "\(tensor) shape \(source.shape) is not affine-Q4 group-64 compatible")
    }

    // Materialize the file-backed BF16 tensor before either conversion timer starts.
    MLX.eval(source)
    Stream.cpu.synchronize()

    let standardStart = ContinuousClock.now
    let standard = MLX.quantized(source, groupSize: 64, bits: 4, mode: .affine)
    guard let standardBiases = standard.biases else {
      throw AuditError.invariant("standard affine quantization omitted biases for \(tensor)")
    }
    MLX.eval(standard.wq, standard.scales, standardBiases)
    Stream.defaultStream(Device.defaultDevice()).synchronize()
    let standardConversionMS = milliseconds(standardStart.duration(to: .now))

    let searchedStart = ContinuousClock.now
    let searched = q4AffineScaleSearchQuantized(source)
    MLX.eval(searched.weight, searched.scales, searched.biases)
    Stream.defaultStream(Device.defaultDevice()).synchronize()
    let searchedConversionMS = milliseconds(searchedStart.duration(to: .now))

    let original = source.asType(.float32)
    let standardRestored = MLX.dequantized(
      standard.wq,
      scales: standard.scales,
      biases: standardBiases,
      groupSize: 64,
      bits: 4,
      mode: .affine,
      dtype: .float32
    )
    let searchedRestored = MLX.dequantized(
      searched.weight,
      scales: searched.scales,
      biases: searched.biases,
      groupSize: 64,
      bits: 4,
      mode: .affine,
      dtype: .float32
    )
    let standardMSE = MLX.mean(MLX.square(original - standardRestored))
    let searchedMSE = MLX.mean(MLX.square(original - searchedRestored))
    let signalMSE = MLX.mean(MLX.square(original))
    let changed = logicalOr(
      searched.scales .!= standard.scales,
      searched.biases .!= standardBiases
    ).asType(.int32).sum()
    MLX.eval(standardMSE, searchedMSE, signalMSE, changed)
    Stream.defaultStream(Device.defaultDevice()).synchronize()

    // MLX's CUDA backend intentionally does not implement float64.  Extract the
    // already-reduced GPU scalars in their native float32/int32 representation,
    // then widen in Swift for stable JSON aggregation.
    let standardMSEValue = Double(standardMSE.item(Float.self))
    let searchedMSEValue = Double(searchedMSE.item(Float.self))
    let signalMSEValue = Double(signalMSE.item(Float.self))
    let changedGroups = Int64(changed.item(Int32.self))
    let groups = Int64(standard.scales.size)
    let standardBytes = storedBytes(
      weight: standard.wq, scales: standard.scales, biases: standardBiases)
    let searchedBytes = storedBytes(
      weight: searched.weight, scales: searched.scales, biases: searched.biases)

    guard standardBytes == searchedBytes else {
      throw AuditError.invariant("searched storage changed for \(tensor)")
    }
    let tolerance = max(1e-12, standardMSEValue * 1e-6)
    guard searchedMSEValue <= standardMSEValue + tolerance else {
      throw AuditError.invariant("searched MSE increased for \(tensor)")
    }
    let reduction = percentReduction(from: standardMSEValue, to: searchedMSEValue)
    let standardRelative = signalMSEValue > 0 ? standardMSEValue / signalMSEValue : 0
    let searchedRelative = signalMSEValue > 0 ? searchedMSEValue / signalMSEValue : 0

    return TensorAuditResult(
      tensor: tensor,
      shard: shard,
      shape: source.shape,
      dtype: String(describing: source.dtype),
      elements: Int64(source.size),
      groups: groups,
      changedGroups: changedGroups,
      changedGroupFraction: groups > 0 ? Double(changedGroups) / Double(groups) : 0,
      standardMSE: standardMSEValue,
      searchedMSE: searchedMSEValue,
      standardRelativeMSE: standardRelative,
      searchedRelativeMSE: searchedRelative,
      mseReductionPercent: reduction,
      standardStoredBytes: standardBytes,
      searchedStoredBytes: searchedBytes,
      standardConversionMS: standardConversionMS,
      searchedConversionMS: searchedConversionMS
    )
  }

  private func aggregate(_ results: [TensorAuditResult]) throws -> AuditTotals {
    guard !results.isEmpty else {
      throw AuditError.invalidInput("no tensors were audited")
    }
    var elements: Int64 = 0
    var groups: Int64 = 0
    var changedGroups: Int64 = 0
    var standardSquaredError = 0.0
    var searchedSquaredError = 0.0
    var tensorsImproved = 0
    for result in results {
      elements = try checkedAdd(elements, result.elements)
      groups = try checkedAdd(groups, result.groups)
      changedGroups = try checkedAdd(changedGroups, result.changedGroups)
      standardSquaredError += result.standardMSE * Double(result.elements)
      searchedSquaredError += result.searchedMSE * Double(result.elements)
      if result.searchedMSE < result.standardMSE {
        tensorsImproved += 1
      }
    }
    guard elements > 0, standardSquaredError.isFinite, searchedSquaredError.isFinite else {
      throw AuditError.invariant("aggregate metrics overflowed")
    }
    let standardMSE = standardSquaredError / Double(elements)
    let searchedMSE = searchedSquaredError / Double(elements)
    let reduction = percentReduction(from: standardMSE, to: searchedMSE)
    return AuditTotals(
      tensors: results.count,
      elements: elements,
      groups: groups,
      changedGroups: changedGroups,
      changedGroupFraction: groups > 0 ? Double(changedGroups) / Double(groups) : 0,
      standardMSE: standardMSE,
      searchedMSE: searchedMSE,
      mseReductionPercent: reduction,
      tensorsImproved: tensorsImproved,
      passesFifteenPercentMSEGate: reduction >= 15
    )
  }

  private func storedBytes(
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray
  ) -> Int64 {
    Int64(weight.nbytes) + Int64(scales.nbytes) + Int64(biases.nbytes)
  }

  private func checkedAdd(_ lhs: Int64, _ rhs: Int64) throws -> Int64 {
    let (result, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else {
      throw AuditError.invariant("integer aggregate overflowed")
    }
    return result
  }

  private func percentReduction(from baseline: Double, to candidate: Double) -> Double {
    baseline > 0 ? (1 - candidate / baseline) * 100 : 0
  }

  private func milliseconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds) * 1_000
      + Double(components.attoseconds) / 1_000_000_000_000_000
  }

  private func formatted(_ value: Double) -> String {
    String(format: "%.3f", value)
  }
}

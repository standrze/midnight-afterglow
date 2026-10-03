import ArgumentParser
import Foundation
import QuantizerSupport
import MLX
import MLXLMCommon
import MLXNN
import WickModelSupport
import ScalePlanMLX

private struct Q4R8PolicyDocument: Decodable {
  let format: Int
  let q8Modules: [String]

  enum CodingKeys: String, CodingKey {
    case format
    case q8Modules = "q8_modules"
  }
}

@main
struct LagunaQuantizer: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: CommandLine.arguments.first?.split(separator: "/").last.map(String.init)
      ?? "wick-laguna-quantize",
    abstract: "Create an MLX-native Laguna Q4R8 checkpoint in Swift."
  )

  @Argument(help: "Unquantized Poolside Laguna safetensors directory.")
  var source: String

  @Argument(help: "New destination directory.")
  var destination: String

  @Option(
    name: .customLong("policy"),
    help: "JSON policy containing format=1 and an exact q8_modules array."
  )
  var policyPath: String?

  @Option(
    name: .customLong("q8-module"),
    help: "Exact module path to promote to Q8; repeat for additional modules."
  )
  var commandLineQ8Modules: [String] = []

  @Flag(
    name: .customLong("q4-scale-search"),
    help: "Experimentally search nearby affine Q4 scales; Q8 modules remain standard affine Q8."
  )
  var q4ScaleSearch = false

  @Option(
    name: .customLong("scale-plan"),
    help: "ScalePlan Q4R8 bundle whose selected per-layer calibrations and Q8 promotions should be applied."
  )
  var scalePlanPath: String?

  @Option(
    name: .customLong("scale-plan-profile"),
    help: "ScalePlan profile: decode-first or prefill-first."
  )
  var scalePlanProfile = "decode-first"

  @Option(
    name: .customLong("max-shard-gib"),
    help: "Maximum output safetensors shard size in GiB."
  )
  var maximumShardGiB: Double = 5

  @Flag(help: "Resolve and print the policy without loading or writing weights.")
  var dryRun = false

  @Flag(
    help: "Evaluate conversion on CPU memory; use when the source model exceeds GPU memory."
  )
  var cpu = false

  @Flag(help: "Replace an existing destination directory.")
  var overwrite = false

  mutating func validate() throws {
    _ = try QuantizerShardSize.bytes(fromGiB: maximumShardGiB)
    guard ["decode-first", "prefill-first"].contains(scalePlanProfile) else {
      throw ValidationError("--scale-plan-profile must be decode-first or prefill-first.")
    }
    if scalePlanPath == nil, scalePlanProfile != "decode-first" {
      throw ValidationError("--scale-plan-profile requires --scale-plan.")
    }
  }

  mutating func run() throws {
    #if os(Linux)
      // MLX-CUDA owns command encoders per thread. Destroy them before the
      // process begins unloading CUDA; otherwise an otherwise successful CLI
      // can abort from a late cudaStreamSynchronize.
      defer { clearStreams() }
    #endif

    if dryRun || cpu {
      // Constructing a model initializes lazy parameter arrays. Keep policy
      // inspection on CPU, and allow large conversions to use system memory
      // when the source checkpoint is larger than discrete GPU memory.
      try Device.withDefaultDevice(.cpu) {
        try execute()
      }
    } else {
      try execute()
    }
  }

  private func execute() throws {
    let sourceURL = URL(fileURLWithPath: source).standardizedFileURL
    let destinationURL = URL(fileURLWithPath: destination).standardizedFileURL
    try QuantizerOutputTransaction.validate(
      sourceDirectory: sourceURL, destinationDirectory: destinationURL, overwrite: overwrite)
    let configURL = sourceURL.appendingPathComponent("config.json")
    let configuration = try JSONDecoder().decode(
      LagunaConfiguration.self,
      from: Data(contentsOf: configURL)
    )
    let model = LagunaModel(configuration)

    let quantizablePaths = Set(
      model.leafModules().flattened().compactMap { path, module in
        module is Quantizable ? path : nil
      }
    )
    let mandatoryRouterPaths = Set(
      quantizablePaths.filter(Self.isRouterProjection)
    )
    let plannedQuantization = try loadScalePlan(
      quantizablePaths: quantizablePaths,
      mandatoryRouterPaths: mandatoryRouterPaths
    )

    var requested = Set(commandLineQ8Modules.map(Self.canonicalModulePath))
    if let policyPath {
      let policyURL = URL(fileURLWithPath: policyPath).standardizedFileURL
      let policy = try JSONDecoder().decode(
        Q4R8PolicyDocument.self,
        from: Data(contentsOf: policyURL)
      )
      guard policy.format == 1 else {
        throw ValidationError("Unsupported Q4R8 policy format \(policy.format); expected 1.")
      }
      requested.formUnion(policy.q8Modules.map(Self.canonicalModulePath))
    }

    let unknown = requested.subtracting(quantizablePaths).sorted()
    guard unknown.isEmpty else {
      throw ValidationError(
        "Q8 policy names unknown or non-quantizable modules: \(unknown.joined(separator: ", "))"
      )
    }

    let planQ8Paths = Set(
      plannedQuantization.compactMap { path, quantization in
        quantization.bits == 8 ? path : nil
      }
    )
    let q8Paths = requested.union(mandatoryRouterPaths).union(planQ8Paths)
    let q4Calibration: ModelConversionQuantizationCalibration =
      q4ScaleSearch ? .q4AffineScaleSearch : .standard
    print(
      "Laguna MLX Q4R8 policy: affine Q4 group-64 default"
        + (q4ScaleSearch ? " with experimental affine scale search" : "")
    )
    if !plannedQuantization.isEmpty {
      print(
        "ScalePlan \(scalePlanProfile): \(plannedQuantization.count) planned units, "
          + "\(planQ8Paths.count) Q8 promotions"
      )
      print(ScalePlanner.experimentalNotice)
    }
    print("Q8 modules: \(q8Paths.count) (\(mandatoryRouterPaths.count) mandatory routers)")
    for path in q8Paths.sorted() {
      print("  \(path)")
    }
    if dryRun {
      return
    }
    print("Conversion device: \(cpu ? "CPU" : String(describing: Device.defaultDevice()))")

    let transaction = try QuantizerOutputTransaction(
      sourceDirectory: sourceURL, destinationDirectory: destinationURL, overwrite: overwrite)
    defer { transaction.cleanup() }
    let maximumShardSize = try QuantizerShardSize.bytes(fromGiB: maximumShardGiB)
    let q8 = ModelConversionQuantization(
      bits: 8,
      groupSize: 64,
      mode: .affine
    )
    let options = ModelConversionOptions(
      bits: 4,
      groupSize: 64,
      mode: .affine,
      calibration: q4Calibration,
      maxShardSize: maximumShardSize,
      overwriteExistingOutput: false,
      quantizationPredicate: { path, _ in
        if q8Paths.contains(path) {
          return .quantize(q8)
        }
        if let planned = plannedQuantization[path] {
          return .quantize(planned)
        }
        return .quantize()
      }
    )

    let result = try convert(
      modelDirectory: sourceURL,
      model: model,
      to: transaction.stagingDirectory,
      options: options,
      progressHandler: { progress in
        let detail = progress.message.map { ": \($0)" } ?? ""
        print("[\(progress.stage.rawValue)]\(detail)")
      }
    )
    if let backup = try transaction.commit() {
      print("Previous output retained at \(backup.path); it could not be removed after publication.")
    }
    print("Created \(destinationURL.path) with \(result.weightsURLs.count) weight shard(s).")
  }

  private static func isRouterProjection(_ path: String) -> Bool {
    path.hasSuffix(".mlp.gate.proj")
  }

  private func loadScalePlan(
    quantizablePaths: Set<String>,
    mandatoryRouterPaths: Set<String>
  ) throws -> [String: ModelConversionQuantization] {
    guard let scalePlanPath else { return [:] }

    let url = URL(fileURLWithPath: scalePlanPath).standardizedFileURL
    let bundle = try JSONDecoder().decode(
      ScalePlanBundle.self,
      from: Data(contentsOf: url)
    )
    guard bundle.formatVersion == 1 else {
      throw ValidationError(
        "Unsupported ScalePlan format \(bundle.formatVersion); expected 1.")
    }
    guard bundle.policy == .q4r8AffineGroup64 else {
      throw ValidationError("The Laguna converter only accepts Q4R8 ScalePlan bundles.")
    }
    let profile =
      scalePlanProfile == "prefill-first" ? bundle.prefillFirst : bundle.decodeFirst
    let expectedObjective: ScalePlanObjective =
      scalePlanProfile == "prefill-first" ? .prefillFirst : .decodeFirst
    guard profile.objective == expectedObjective else {
      throw ValidationError(
        "ScalePlan \(scalePlanProfile) profile is labeled \(profile.objective.rawValue).")
    }

    var result: [String: ModelConversionQuantization] = [:]
    for selection in profile.selections {
      let path = Self.canonicalModulePath(selection.layer)
      guard result[path] == nil else {
        throw ValidationError(
          "ScalePlan has duplicate selections for canonical module \(path).")
      }
      guard quantizablePaths.contains(path) else {
        throw ValidationError(
          "ScalePlan names unknown or non-quantizable module: \(selection.layer)")
      }
      let candidate = selection.candidate
      guard candidate.format == .affine,
        candidate.group == 64,
        candidate.bits == 4 || candidate.bits == 8
      else {
        throw ValidationError(
          "ScalePlan candidate \(candidate.id) for \(selection.layer) is not affine Q4/Q8 group-64."
        )
      }
      if mandatoryRouterPaths.contains(path), candidate.bits != 8 {
        throw ValidationError("ScalePlan must keep mandatory router \(selection.layer) in Q8.")
      }
      let calibration: ModelConversionQuantizationCalibration
      switch candidate.calibration {
      case .standard:
        calibration = .standard
      case .q4AffineScaleSearch:
        guard candidate.bits == 4 else {
          throw ValidationError(
            "ScalePlan applies Q4 affine scale search to non-Q4 candidate \(candidate.id).")
        }
        calibration = .q4AffineScaleSearch
      }
      let quantization = ModelConversionQuantization(
        bits: candidate.bits,
        groupSize: candidate.group,
        mode: .affine,
        calibration: calibration
      )
      result[path] = quantization
    }
    return result
  }

  private static func canonicalModulePath(_ input: String) -> String {
    var path = input.trimmingCharacters(in: .whitespacesAndNewlines)
    if path.hasPrefix("model.") || path.hasPrefix("lm_head") {
      path = "language_model.\(path)"
    }
    path = path.replacingOccurrences(
      of: ".switch_mlp.gate_proj",
      with: ".switch_mlp.gate_up_proj"
    )
    path = path.replacingOccurrences(
      of: ".switch_mlp.up_proj",
      with: ".switch_mlp.gate_up_proj"
    )
    path = path.replacingOccurrences(
      of: ".shared_expert.gate_proj",
      with: ".shared_expert.gate_up_proj"
    )
    path = path.replacingOccurrences(
      of: ".shared_expert.up_proj",
      with: ".shared_expert.gate_up_proj"
    )
    if path.hasSuffix(".mlp.gate_proj") {
      path = String(path.dropLast("gate_proj".count)) + "gate_up_proj"
    } else if path.hasSuffix(".mlp.up_proj") {
      path = String(path.dropLast("up_proj".count)) + "gate_up_proj"
    }
    return path
  }
}

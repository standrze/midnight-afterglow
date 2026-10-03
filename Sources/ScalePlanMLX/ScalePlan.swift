import Foundation

/// Maturity attached to measured inputs and generated plans.
///
/// ScalePlan-MLX remains a research hypothesis until the report's model-quality,
/// same-format latency, and multi-chip acceptance gates have all passed.
public enum ScalePlanStatus: String, Codable, Sendable {
  case experimentalUnbenchmarked = "experimental_unbenchmarked"
  case measuredCandidate = "measured_candidate"
  case validated = "validated"
}

/// This first implementation deliberately preserves the runner's established
/// Q4R8 deployment policy. Other formats require a separate explicit experiment.
public enum ScalePlanPolicy: String, Codable, Sendable {
  case q4r8AffineGroup64 = "q4r8_affine_group64"
}

public enum ScalePlanQualityProxyKind: String, Codable, Sendable {
  /// HIGGS-style sum of `alpha[layer] * relative_mse[layer, candidate]`.
  case higgsLinearizedMSE = "higgs_linearized_mse"

  /// A measured per-candidate teacher KL proxy whose additive model has been
  /// checked against held-out full-model measurements.
  case teacherKL = "teacher_kl"
}

public enum ScalePlanLayerRole: String, Codable, Sendable {
  case weight
  case router
}

public enum ScalePlanQuantizationFormat: String, Codable, Sendable {
  case affine
}

public enum ScalePlanCalibration: String, Codable, Sendable {
  case standard
  case q4AffineScaleSearch = "q4_affine_scale_search"
}

public enum ScalePlanObjective: String, Codable, Sendable {
  case decodeFirst = "decode_first"
  case prefillFirst = "prefill_first"
}

public struct ScalePlanEnvironment: Codable, Equatable, Sendable {
  public var chip: String
  public var operatingSystem: String
  public var mlxCommit: String
  public var converterCommit: String

  public init(
    chip: String,
    operatingSystem: String,
    mlxCommit: String,
    converterCommit: String
  ) {
    self.chip = chip
    self.operatingSystem = operatingSystem
    self.mlxCommit = mlxCommit
    self.converterCommit = converterCommit
  }

  enum CodingKeys: String, CodingKey {
    case chip
    case operatingSystem = "os"
    case mlxCommit = "mlx_commit"
    case converterCommit = "converter_commit"
  }
}

public struct ScalePlanQualityProxy: Codable, Equatable, Sendable {
  public var kind: ScalePlanQualityProxyKind
  public var heldOutValidated: Bool
  public var validationNotes: String?

  public init(
    kind: ScalePlanQualityProxyKind,
    heldOutValidated: Bool,
    validationNotes: String? = nil
  ) {
    self.kind = kind
    self.heldOutValidated = heldOutValidated
    self.validationNotes = validationNotes
  }

  enum CodingKeys: String, CodingKey {
    case kind
    case heldOutValidated = "held_out_validated"
    case validationNotes = "validation_notes"
  }
}

/// One measured precision choice for one indivisible quantization unit.
///
/// Times are warm medians for the actual layer shape and workload. Payload is
/// the packed weight payload; metadata includes scales, affine biases, and any
/// other stored arrays needed by that candidate.
public struct ScalePlanCandidate: Codable, Equatable, Sendable {
  public var id: String
  public var format: ScalePlanQuantizationFormat
  public var calibration: ScalePlanCalibration
  public var bits: Int
  public var group: Int
  public var payloadBits: Int64
  public var metadataBytes: Int64
  public var reconstructionMSE: Double
  public var relativeMSE: Double
  public var teacherKL: Double?
  public var decodeNS: Int64
  public var verifyNS: Int64
  public var prefillNS: Int64
  public var peakBytes: Int64
  public var contiguityCopyBytes: Int64
  public var coldCompileMS: Double

  public init(
    id: String,
    format: ScalePlanQuantizationFormat = .affine,
    calibration: ScalePlanCalibration = .standard,
    bits: Int,
    group: Int = 64,
    payloadBits: Int64,
    metadataBytes: Int64,
    reconstructionMSE: Double,
    relativeMSE: Double,
    teacherKL: Double? = nil,
    decodeNS: Int64,
    verifyNS: Int64,
    prefillNS: Int64,
    peakBytes: Int64,
    contiguityCopyBytes: Int64,
    coldCompileMS: Double
  ) {
    self.id = id
    self.format = format
    self.calibration = calibration
    self.bits = bits
    self.group = group
    self.payloadBits = payloadBits
    self.metadataBytes = metadataBytes
    self.reconstructionMSE = reconstructionMSE
    self.relativeMSE = relativeMSE
    self.teacherKL = teacherKL
    self.decodeNS = decodeNS
    self.verifyNS = verifyNS
    self.prefillNS = prefillNS
    self.peakBytes = peakBytes
    self.contiguityCopyBytes = contiguityCopyBytes
    self.coldCompileMS = coldCompileMS
  }

  public var storedBytes: Int64 {
    (payloadBits + 7) / 8 + metadataBytes
  }

  enum CodingKeys: String, CodingKey {
    case id, format, calibration, bits, group
    case payloadBits = "payload_bits"
    case metadataBytes = "metadata_bytes"
    case reconstructionMSE = "reconstruction_mse"
    case relativeMSE = "relative_mse"
    case teacherKL = "teacher_kl"
    case decodeNS = "decode_ns"
    case verifyNS = "verify_ns"
    case prefillNS = "prefill_ns"
    case peakBytes = "peak_bytes"
    case contiguityCopyBytes = "contiguity_copy_bytes"
    case coldCompileMS = "cold_compile_ms"
  }
}

public struct ScalePlanLayer: Codable, Equatable, Sendable {
  public var layer: String
  public var shape: [Int]
  public var role: ScalePlanLayerRole
  public var qualityAlpha: Double?
  public var candidates: [ScalePlanCandidate]

  public init(
    layer: String,
    shape: [Int],
    role: ScalePlanLayerRole = .weight,
    qualityAlpha: Double? = nil,
    candidates: [ScalePlanCandidate]
  ) {
    self.layer = layer
    self.shape = shape
    self.role = role
    self.qualityAlpha = qualityAlpha
    self.candidates = candidates
  }

  enum CodingKeys: String, CodingKey {
    case layer, shape, role, candidates
    case qualityAlpha = "quality_alpha"
  }
}

public struct ScalePlanLedger: Codable, Equatable, Sendable {
  public var formatVersion: Int
  public var status: ScalePlanStatus
  public var policy: ScalePlanPolicy
  public var model: String
  public var calibrationSeed: Int
  public var environment: ScalePlanEnvironment
  public var qualityProxy: ScalePlanQualityProxy
  public var layers: [ScalePlanLayer]

  public init(
    formatVersion: Int = 1,
    status: ScalePlanStatus = .experimentalUnbenchmarked,
    policy: ScalePlanPolicy = .q4r8AffineGroup64,
    model: String,
    calibrationSeed: Int,
    environment: ScalePlanEnvironment,
    qualityProxy: ScalePlanQualityProxy,
    layers: [ScalePlanLayer]
  ) {
    self.formatVersion = formatVersion
    self.status = status
    self.policy = policy
    self.model = model
    self.calibrationSeed = calibrationSeed
    self.environment = environment
    self.qualityProxy = qualityProxy
    self.layers = layers
  }

  enum CodingKeys: String, CodingKey {
    case status, policy, model, environment, layers
    case formatVersion = "format"
    case calibrationSeed = "calibration_seed"
    case qualityProxy = "quality_proxy"
  }
}

public struct ScalePlanConstraints: Codable, Equatable, Sendable {
  public var maximumStoredBytes: Int64
  public var maximumLatencyNS: Int64
  public var byteQuantum: Int64
  public var latencyQuantumNS: Int64
  public var maximumStates: Int
  public var allowUnvalidatedProxy: Bool

  public init(
    maximumStoredBytes: Int64,
    maximumLatencyNS: Int64,
    byteQuantum: Int64 = 4_096,
    latencyQuantumNS: Int64 = 100,
    maximumStates: Int = 250_000,
    allowUnvalidatedProxy: Bool = false
  ) {
    self.maximumStoredBytes = maximumStoredBytes
    self.maximumLatencyNS = maximumLatencyNS
    self.byteQuantum = byteQuantum
    self.latencyQuantumNS = latencyQuantumNS
    self.maximumStates = maximumStates
    self.allowUnvalidatedProxy = allowUnvalidatedProxy
  }

  enum CodingKeys: String, CodingKey {
    case maximumStoredBytes = "maximum_stored_bytes"
    case maximumLatencyNS = "maximum_latency_ns"
    case byteQuantum = "byte_quantum"
    case latencyQuantumNS = "latency_quantum_ns"
    case maximumStates = "maximum_states"
    case allowUnvalidatedProxy = "allow_unvalidated_proxy"
  }
}

public struct ScalePlanSelection: Codable, Equatable, Sendable {
  public var layer: String
  public var shape: [Int]
  public var role: ScalePlanLayerRole
  public var candidate: ScalePlanCandidate
  public var predictedQualityLoss: Double

  public init(
    layer: String,
    shape: [Int],
    role: ScalePlanLayerRole,
    candidate: ScalePlanCandidate,
    predictedQualityLoss: Double
  ) {
    self.layer = layer
    self.shape = shape
    self.role = role
    self.candidate = candidate
    self.predictedQualityLoss = predictedQualityLoss
  }

  enum CodingKeys: String, CodingKey {
    case layer, shape, role, candidate
    case predictedQualityLoss = "predicted_quality_loss"
  }
}

public struct ScalePlanTotals: Codable, Equatable, Sendable {
  public var storedBytes: Int64
  public var decodeNS: Int64
  public var verifyNS: Int64
  public var prefillNS: Int64
  public var predictedQualityLoss: Double
  public var peakBytesUpperBound: Int64
  public var contiguityCopyBytes: Int64
  public var coldCompileMS: Double

  enum CodingKeys: String, CodingKey {
    case storedBytes = "stored_bytes"
    case decodeNS = "decode_ns"
    case verifyNS = "verify_ns"
    case prefillNS = "prefill_ns"
    case predictedQualityLoss = "predicted_quality_loss"
    case peakBytesUpperBound = "peak_bytes_upper_bound"
    case contiguityCopyBytes = "contiguity_copy_bytes"
    case coldCompileMS = "cold_compile_ms"
  }
}

public struct ScalePlanProfile: Codable, Equatable, Sendable {
  public var objective: ScalePlanObjective
  public var constraints: ScalePlanConstraints
  public var selections: [ScalePlanSelection]
  public var totals: ScalePlanTotals

  public init(
    objective: ScalePlanObjective,
    constraints: ScalePlanConstraints,
    selections: [ScalePlanSelection],
    totals: ScalePlanTotals
  ) {
    self.objective = objective
    self.constraints = constraints
    self.selections = selections
    self.totals = totals
  }
}

public struct ScalePlanBundle: Codable, Equatable, Sendable {
  public var formatVersion: Int
  public var status: ScalePlanStatus
  public var policy: ScalePlanPolicy
  public var sourceModel: String
  public var environment: ScalePlanEnvironment
  public var qualityProxy: ScalePlanQualityProxy
  public var experimentalNotice: String
  public var decodeFirst: ScalePlanProfile
  public var prefillFirst: ScalePlanProfile

  public init(
    formatVersion: Int = 1,
    status: ScalePlanStatus,
    policy: ScalePlanPolicy,
    sourceModel: String,
    environment: ScalePlanEnvironment,
    qualityProxy: ScalePlanQualityProxy,
    experimentalNotice: String = ScalePlanner.experimentalNotice,
    decodeFirst: ScalePlanProfile,
    prefillFirst: ScalePlanProfile
  ) {
    self.formatVersion = formatVersion
    self.status = status
    self.policy = policy
    self.sourceModel = sourceModel
    self.environment = environment
    self.qualityProxy = qualityProxy
    self.experimentalNotice = experimentalNotice
    self.decodeFirst = decodeFirst
    self.prefillFirst = prefillFirst
  }

  enum CodingKeys: String, CodingKey {
    case status, policy, environment
    case formatVersion = "format"
    case sourceModel = "source_model"
    case qualityProxy = "quality_proxy"
    case experimentalNotice = "experimental_notice"
    case decodeFirst = "decode_first"
    case prefillFirst = "prefill_first"
  }
}

public enum ScalePlanError: Error, LocalizedError, Equatable {
  case invalidLedger(String)
  case invalidConstraints(String)
  case unvalidatedQualityProxy(String)
  case noFeasiblePlan(ScalePlanObjective)
  case stateLimitExceeded(layer: String, limit: Int)

  public var errorDescription: String? {
    switch self {
    case .invalidLedger(let message):
      return "Invalid ScalePlan ledger: \(message)"
    case .invalidConstraints(let message):
      return "Invalid ScalePlan constraints: \(message)"
    case .unvalidatedQualityProxy(let message):
      return "ScalePlan quality proxy is not held-out validated: \(message)"
    case .noFeasiblePlan(let objective):
      return "No feasible \(objective.rawValue) plan fits both budgets."
    case .stateLimitExceeded(let layer, let limit):
      return "ScalePlan exceeded its \(limit)-state safety limit while adding \(layer). Increase the cost quanta or explicitly raise the state limit."
    }
  }
}

/// Multiple-choice, budget-constrained per-layer planner.
///
/// Costs are rounded upward into explicit byte and latency quanta, making the
/// dynamic program conservative at the budget boundary. It keeps only the
/// lowest predicted loss for each quantized cost pair and fails closed rather
/// than silently truncating a large search frontier.
public struct ScalePlanner: Sendable {
  public static let experimentalNotice =
    "ScalePlan-MLX is an experimental research hypothesis. This plan is not a performance or quality claim until its held-out and hardware acceptance gates pass."

  public init() {}

  public func makeBundle(
    ledger: ScalePlanLedger,
    decodeConstraints: ScalePlanConstraints,
    prefillConstraints: ScalePlanConstraints
  ) throws -> ScalePlanBundle {
    let decode = try plan(
      ledger: ledger, objective: .decodeFirst, constraints: decodeConstraints)
    let prefill = try plan(
      ledger: ledger, objective: .prefillFirst, constraints: prefillConstraints)
    return ScalePlanBundle(
      status: ledger.status,
      policy: ledger.policy,
      sourceModel: ledger.model,
      environment: ledger.environment,
      qualityProxy: ledger.qualityProxy,
      decodeFirst: decode,
      prefillFirst: prefill
    )
  }

  public func plan(
    ledger: ScalePlanLedger,
    objective: ScalePlanObjective,
    constraints: ScalePlanConstraints
  ) throws -> ScalePlanProfile {
    try validate(ledger: ledger, constraints: constraints)

    let byteBudget = constraints.maximumStoredBytes / constraints.byteQuantum
    let latencyBudget = constraints.maximumLatencyNS / constraints.latencyQuantumNS
    var states: [CostKey: State] = [.init(bytes: 0, latency: 0): .init(loss: 0, choices: [])]

    for layer in ledger.layers {
      let candidates = eligibleCandidates(for: layer).sorted { $0.id < $1.id }
      guard !candidates.isEmpty else {
        throw ScalePlanError.invalidLedger(
          "layer \(layer.layer) has no candidate allowed by the Q4R8/router policy")
      }

      var next: [CostKey: State] = [:]
      for (key, state) in states.sorted(by: stateOrder) {
        for candidate in candidates {
          let bytes = try units(candidate.storedBytes, quantum: constraints.byteQuantum)
          let latencyValue = latency(of: candidate, objective: objective)
          let latency = try units(latencyValue, quantum: constraints.latencyQuantumNS)
          let (newBytes, bytesOverflow) = key.bytes.addingReportingOverflow(bytes)
          let (newLatency, latencyOverflow) = key.latency.addingReportingOverflow(latency)
          if bytesOverflow || latencyOverflow || newBytes > byteBudget || newLatency > latencyBudget {
            continue
          }

          let candidateLoss = try predictedLoss(
            candidate: candidate, layer: layer, proxy: ledger.qualityProxy.kind)
          let newLoss = state.loss + candidateLoss
          guard newLoss.isFinite else {
            throw ScalePlanError.invalidLedger(
              "predicted loss overflowed while adding \(layer.layer)/\(candidate.id)")
          }

          let newKey = CostKey(bytes: newBytes, latency: newLatency)
          if let existing = next[newKey], existing.loss <= newLoss {
            continue
          }
          next[newKey] = State(
            loss: newLoss,
            choices: state.choices + [
              Choice(candidate: candidate, predictedLoss: candidateLoss)
            ]
          )
          if next.count > constraints.maximumStates {
            throw ScalePlanError.stateLimitExceeded(
              layer: layer.layer, limit: constraints.maximumStates)
          }
        }
      }
      guard !next.isEmpty else {
        throw ScalePlanError.noFeasiblePlan(objective)
      }
      states = next
    }

    guard let winner = states.min(by: winningStateOrder)?.value else {
      throw ScalePlanError.noFeasiblePlan(objective)
    }
    let selections = zip(ledger.layers, winner.choices).map { layer, choice in
      ScalePlanSelection(
        layer: layer.layer,
        shape: layer.shape,
        role: layer.role,
        candidate: choice.candidate,
        predictedQualityLoss: choice.predictedLoss
      )
    }
    let totals = try totals(for: selections)
    guard totals.storedBytes <= constraints.maximumStoredBytes,
      latency(of: totals, objective: objective) <= constraints.maximumLatencyNS
    else {
      throw ScalePlanError.invalidLedger(
        "internal conservative-budget invariant failed for \(objective.rawValue)")
    }
    return ScalePlanProfile(
      objective: objective,
      constraints: constraints,
      selections: selections,
      totals: totals
    )
  }

  private func validate(
    ledger: ScalePlanLedger,
    constraints: ScalePlanConstraints
  ) throws {
    guard ledger.formatVersion == 1 else {
      throw ScalePlanError.invalidLedger(
        "unsupported format \(ledger.formatVersion); expected 1")
    }
    guard ledger.policy == .q4r8AffineGroup64 else {
      throw ScalePlanError.invalidLedger("this implementation only accepts Q4R8")
    }
    guard !ledger.model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw ScalePlanError.invalidLedger("model must not be empty")
    }
    guard !ledger.layers.isEmpty else {
      throw ScalePlanError.invalidLedger("at least one layer is required")
    }
    guard constraints.maximumStoredBytes > 0,
      constraints.maximumLatencyNS > 0,
      constraints.byteQuantum > 0,
      constraints.latencyQuantumNS > 0,
      constraints.maximumStates > 0
    else {
      throw ScalePlanError.invalidConstraints("budgets, quanta, and state limit must be positive")
    }
    if !ledger.qualityProxy.heldOutValidated && !constraints.allowUnvalidatedProxy {
      throw ScalePlanError.unvalidatedQualityProxy(
        ledger.qualityProxy.validationNotes ?? "no validation notes supplied")
    }

    let environmentValues = [
      ledger.environment.chip,
      ledger.environment.operatingSystem,
      ledger.environment.mlxCommit,
      ledger.environment.converterCommit,
    ]
    guard environmentValues.allSatisfy({
      !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }) else {
      throw ScalePlanError.invalidLedger(
        "chip, os, mlx_commit, and converter_commit are all required")
    }

    var layerNames = Set<String>()
    for layer in ledger.layers {
      let normalizedLayerName = layer.layer.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !normalizedLayerName.isEmpty,
        layerNames.insert(normalizedLayerName).inserted
      else {
        throw ScalePlanError.invalidLedger("empty or duplicate layer \(layer.layer)")
      }
      guard !layer.shape.isEmpty, layer.shape.allSatisfy({ $0 > 0 }) else {
        throw ScalePlanError.invalidLedger("layer \(layer.layer) has an invalid name or shape")
      }
      guard !layer.candidates.isEmpty else {
        throw ScalePlanError.invalidLedger("layer \(layer.layer) has no candidates")
      }
      var candidateIDs = Set<String>()
      for candidate in layer.candidates {
        let normalizedCandidateID = candidate.id.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedCandidateID.isEmpty,
          candidateIDs.insert(normalizedCandidateID).inserted
        else {
          throw ScalePlanError.invalidLedger(
            "layer \(layer.layer) has an empty or duplicate candidate id \(candidate.id)")
        }
        try validate(candidate: candidate, layer: layer, proxy: ledger.qualityProxy.kind)
      }
    }
  }

  private func validate(
    candidate: ScalePlanCandidate,
    layer: ScalePlanLayer,
    proxy: ScalePlanQualityProxyKind
  ) throws {
    guard candidate.format == .affine,
      candidate.group == 64,
      candidate.bits == 4 || candidate.bits == 8
    else {
      throw ScalePlanError.invalidLedger(
        "\(layer.layer)/\(candidate.id) is not affine Q4/Q8 group-64")
    }
    guard candidate.bits == 4 || candidate.calibration == .standard else {
      throw ScalePlanError.invalidLedger(
        "\(layer.layer)/\(candidate.id) applies Q4 scale search to a Q8 candidate")
    }
    guard candidate.payloadBits >= 0, candidate.payloadBits <= Int64.max - 7,
      candidate.metadataBytes >= 0,
      candidate.metadataBytes <= Int64.max - ((candidate.payloadBits + 7) / 8),
      candidate.decodeNS >= 0,
      candidate.verifyNS >= 0,
      candidate.prefillNS >= 0,
      candidate.peakBytes >= 0,
      candidate.contiguityCopyBytes >= 0
    else {
      throw ScalePlanError.invalidLedger(
        "\(layer.layer)/\(candidate.id) has a negative or overflowing cost")
    }
    let finiteMetrics = [
      candidate.reconstructionMSE,
      candidate.relativeMSE,
      candidate.coldCompileMS,
    ]
    guard finiteMetrics.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
      throw ScalePlanError.invalidLedger(
        "\(layer.layer)/\(candidate.id) has a non-finite or negative metric")
    }
    if let teacherKL = candidate.teacherKL,
      !teacherKL.isFinite || teacherKL < 0
    {
      throw ScalePlanError.invalidLedger(
        "\(layer.layer)/\(candidate.id) has a non-finite or negative teacher_kl")
    }
    if let qualityAlpha = layer.qualityAlpha,
      !qualityAlpha.isFinite || qualityAlpha < 0
    {
      throw ScalePlanError.invalidLedger(
        "\(layer.layer) has a non-finite or negative quality_alpha")
    }
    switch proxy {
    case .higgsLinearizedMSE:
      guard let alpha = layer.qualityAlpha, alpha.isFinite, alpha >= 0 else {
        throw ScalePlanError.invalidLedger(
          "\(layer.layer) requires a finite non-negative quality_alpha for the HIGGS proxy")
      }
    case .teacherKL:
      guard let kl = candidate.teacherKL, kl.isFinite, kl >= 0 else {
        throw ScalePlanError.invalidLedger(
          "\(layer.layer)/\(candidate.id) requires finite non-negative teacher_kl")
      }
    }
  }

  private func eligibleCandidates(for layer: ScalePlanLayer) -> [ScalePlanCandidate] {
    if layer.role == .router {
      return layer.candidates.filter { $0.bits == 8 }
    }
    return layer.candidates
  }

  private func predictedLoss(
    candidate: ScalePlanCandidate,
    layer: ScalePlanLayer,
    proxy: ScalePlanQualityProxyKind
  ) throws -> Double {
    switch proxy {
    case .higgsLinearizedMSE:
      guard let alpha = layer.qualityAlpha else {
        throw ScalePlanError.invalidLedger("missing quality_alpha for \(layer.layer)")
      }
      return alpha * candidate.relativeMSE
    case .teacherKL:
      guard let teacherKL = candidate.teacherKL else {
        throw ScalePlanError.invalidLedger(
          "missing teacher_kl for \(layer.layer)/\(candidate.id)")
      }
      return teacherKL
    }
  }

  private func units(_ value: Int64, quantum: Int64) throws -> Int64 {
    guard value <= Int64.max - (quantum - 1) else {
      throw ScalePlanError.invalidLedger("cost cannot be rounded without overflowing")
    }
    return (value + quantum - 1) / quantum
  }

  private func latency(
    of candidate: ScalePlanCandidate,
    objective: ScalePlanObjective
  ) -> Int64 {
    switch objective {
    case .decodeFirst: candidate.decodeNS
    case .prefillFirst: candidate.prefillNS
    }
  }

  private func latency(of totals: ScalePlanTotals, objective: ScalePlanObjective) -> Int64 {
    switch objective {
    case .decodeFirst: totals.decodeNS
    case .prefillFirst: totals.prefillNS
    }
  }

  private func totals(for selections: [ScalePlanSelection]) throws -> ScalePlanTotals {
    var storedBytes: Int64 = 0
    var decodeNS: Int64 = 0
    var verifyNS: Int64 = 0
    var prefillNS: Int64 = 0
    var qualityLoss = 0.0
    var peakBytes: Int64 = 0
    var copyBytes: Int64 = 0
    var coldCompileMS = 0.0

    for selection in selections {
      storedBytes = try checkedAdd(storedBytes, selection.candidate.storedBytes)
      decodeNS = try checkedAdd(decodeNS, selection.candidate.decodeNS)
      verifyNS = try checkedAdd(verifyNS, selection.candidate.verifyNS)
      prefillNS = try checkedAdd(prefillNS, selection.candidate.prefillNS)
      peakBytes = try checkedAdd(peakBytes, selection.candidate.peakBytes)
      copyBytes = try checkedAdd(copyBytes, selection.candidate.contiguityCopyBytes)
      qualityLoss += selection.predictedQualityLoss
      coldCompileMS += selection.candidate.coldCompileMS
    }
    guard qualityLoss.isFinite, coldCompileMS.isFinite else {
      throw ScalePlanError.invalidLedger("aggregate floating-point metric overflowed")
    }
    return ScalePlanTotals(
      storedBytes: storedBytes,
      decodeNS: decodeNS,
      verifyNS: verifyNS,
      prefillNS: prefillNS,
      predictedQualityLoss: qualityLoss,
      peakBytesUpperBound: peakBytes,
      contiguityCopyBytes: copyBytes,
      coldCompileMS: coldCompileMS
    )
  }

  private func checkedAdd(_ lhs: Int64, _ rhs: Int64) throws -> Int64 {
    let (result, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else {
      throw ScalePlanError.invalidLedger("aggregate cost overflowed Int64")
    }
    return result
  }
}

private struct CostKey: Hashable {
  var bytes: Int64
  var latency: Int64
}

private struct Choice {
  var candidate: ScalePlanCandidate
  var predictedLoss: Double
}

private struct State {
  var loss: Double
  var choices: [Choice]
}

private func stateOrder(
  _ lhs: Dictionary<CostKey, State>.Element,
  _ rhs: Dictionary<CostKey, State>.Element
) -> Bool {
  if lhs.key.bytes != rhs.key.bytes { return lhs.key.bytes < rhs.key.bytes }
  return lhs.key.latency < rhs.key.latency
}

private func winningStateOrder(
  _ lhs: Dictionary<CostKey, State>.Element,
  _ rhs: Dictionary<CostKey, State>.Element
) -> Bool {
  if lhs.value.loss != rhs.value.loss { return lhs.value.loss < rhs.value.loss }
  if lhs.key.latency != rhs.key.latency { return lhs.key.latency < rhs.key.latency }
  return lhs.key.bytes < rhs.key.bytes
}

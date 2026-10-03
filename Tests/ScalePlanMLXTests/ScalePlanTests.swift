import Foundation
import Testing

@testable import ScalePlanMLX

@Suite("ScalePlan Q4R8 planner")
struct ScalePlanTests {
  @Test("Unvalidated quality forecasts fail closed by default")
  func rejectsUnvalidatedProxy() {
    let ledger = makeLedger(heldOutValidated: false)
    let constraints = makeConstraints()

    #expect(throws: ScalePlanError.self) {
      try ScalePlanner().plan(
        ledger: ledger,
        objective: .decodeFirst,
        constraints: constraints
      )
    }
  }

  @Test("An explicit plumbing override retains the experimental status")
  func permitsExplicitUnvalidatedExperiment() throws {
    let ledger = makeLedger(heldOutValidated: false)
    var constraints = makeConstraints()
    constraints.allowUnvalidatedProxy = true

    let profile = try ScalePlanner().plan(
      ledger: ledger,
      objective: .decodeFirst,
      constraints: constraints
    )

    #expect(profile.selections.count == ledger.layers.count)
  }

  @Test("Routers remain Q8 even when a lower-loss Q4 row is present")
  func forcesRoutersToQ8() throws {
    let profile = try ScalePlanner().plan(
      ledger: makeLedger(),
      objective: .decodeFirst,
      constraints: makeConstraints()
    )

    let router = try #require(profile.selections.first { $0.role == .router })
    #expect(router.candidate.bits == 8)
    #expect(router.candidate.id == "router-q8")
  }

  @Test("Affine scale search can compete as a same-cost Q4 calibration")
  func selectsScaleSearchedQ4AtMatchedCost() throws {
    var ledger = makeLedger()
    ledger.layers[0].candidates.append(
      candidate(
        id: "weight-q4-scale-search",
        calibration: .q4AffineScaleSearch,
        bits: 4,
        bytes: 100,
        relativeMSE: 0.45,
        decodeNS: 10,
        prefillNS: 50
      ))
    var constraints = makeConstraints()
    constraints.maximumStoredBytes = 300

    let profile = try ScalePlanner().plan(
      ledger: ledger,
      objective: .decodeFirst,
      constraints: constraints
    )

    #expect(profile.selections[0].candidate.id == "weight-q4-scale-search")
    #expect(profile.selections[0].candidate.bits == 4)
  }

  @Test("Decode and prefill budgets can produce different plans")
  func producesSeparateWorkloadPlans() throws {
    let ledger = makeLedger()
    let planner = ScalePlanner()
    let decode = try planner.plan(
      ledger: ledger,
      objective: .decodeFirst,
      constraints: ScalePlanConstraints(
        maximumStoredBytes: 400,
        maximumLatencyNS: 35,
        byteQuantum: 1,
        latencyQuantumNS: 1
      )
    )
    let prefill = try planner.plan(
      ledger: ledger,
      objective: .prefillFirst,
      constraints: ScalePlanConstraints(
        maximumStoredBytes: 400,
        maximumLatencyNS: 75,
        byteQuantum: 1,
        latencyQuantumNS: 1
      )
    )

    #expect(decode.selections[0].candidate.bits == 4)
    #expect(prefill.selections[0].candidate.bits == 8)
  }

  @Test("The implementation rejects candidates outside Q4R8")
  func rejectsNonQ4R8Geometry() {
    var ledger = makeLedger()
    ledger.layers[0].candidates[0].group = 128

    #expect(throws: ScalePlanError.self) {
      try ScalePlanner().plan(
        ledger: ledger,
        objective: .decodeFirst,
        constraints: makeConstraints()
      )
    }
  }

  @Test("Scale search cannot be attached to a Q8 candidate")
  func rejectsScaleSearchOnQ8() {
    var ledger = makeLedger()
    ledger.layers[0].candidates[1].calibration = .q4AffineScaleSearch

    #expect(throws: ScalePlanError.self) {
      try ScalePlanner().plan(
        ledger: ledger,
        objective: .decodeFirst,
        constraints: makeConstraints()
      )
    }
  }

  @Test("Whitespace-only and trim-colliding identifiers fail closed")
  func rejectsAmbiguousIdentifiers() {
    var blank = makeLedger()
    blank.layers[0].candidates[0].id = " \n "
    #expect(throws: ScalePlanError.self) {
      try ScalePlanner().plan(
        ledger: blank,
        objective: .decodeFirst,
        constraints: makeConstraints()
      )
    }

    var collision = makeLedger()
    collision.layers[1].layer = " \(collision.layers[0].layer) "
    #expect(throws: ScalePlanError.self) {
      try ScalePlanner().plan(
        ledger: collision,
        objective: .decodeFirst,
        constraints: makeConstraints()
      )
    }
  }

  @Test("Optional metrics are validated even when another proxy is selected")
  func rejectsInvalidOptionalMetrics() {
    var ledger = makeLedger()
    ledger.layers[0].candidates[0].teacherKL = .infinity

    #expect(throws: ScalePlanError.self) {
      try ScalePlanner().plan(
        ledger: ledger,
        objective: .decodeFirst,
        constraints: makeConstraints()
      )
    }
  }

  @Test("A generated bundle round-trips through its audited JSON schema")
  func bundleRoundTrip() throws {
    let ledger = makeLedger()
    let constraints = makeConstraints()
    let bundle = try ScalePlanner().makeBundle(
      ledger: ledger,
      decodeConstraints: constraints,
      prefillConstraints: constraints
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let encoded = try encoder.encode(bundle)
    let decoded = try JSONDecoder().decode(ScalePlanBundle.self, from: encoded)

    #expect(decoded == bundle)
    #expect(String(decoding: encoded, as: UTF8.self).contains("experimental_notice"))
  }

  private func makeConstraints() -> ScalePlanConstraints {
    ScalePlanConstraints(
      maximumStoredBytes: 400,
      maximumLatencyNS: 1_000,
      byteQuantum: 1,
      latencyQuantumNS: 1
    )
  }

  private func makeLedger(heldOutValidated: Bool = true) -> ScalePlanLedger {
    ScalePlanLedger(
      model: "laguna-test",
      calibrationSeed: 7,
      environment: ScalePlanEnvironment(
        chip: "test-chip",
        operatingSystem: "test-os",
        mlxCommit: "mlx-test",
        converterCommit: "converter-test"
      ),
      qualityProxy: ScalePlanQualityProxy(
        kind: .higgsLinearizedMSE,
        heldOutValidated: heldOutValidated,
        validationNotes: heldOutValidated ? "held-out forecast passed" : "not measured"
      ),
      layers: [
        ScalePlanLayer(
          layer: "language_model.model.layers.0.self_attn.q_proj",
          shape: [64, 64],
          qualityAlpha: 1,
          candidates: [
            candidate(
              id: "weight-q4",
              bits: 4,
              bytes: 100,
              relativeMSE: 1,
              decodeNS: 10,
              prefillNS: 50
            ),
            candidate(
              id: "weight-q8",
              bits: 8,
              bytes: 200,
              relativeMSE: 0.1,
              decodeNS: 30,
              prefillNS: 20
            ),
          ]
        ),
        ScalePlanLayer(
          layer: "language_model.model.layers.1.mlp.gate.proj",
          shape: [256, 2_048],
          role: .router,
          qualityAlpha: 1,
          candidates: [
            candidate(
              id: "router-q4",
              bits: 4,
              bytes: 100,
              relativeMSE: 0,
              decodeNS: 5,
              prefillNS: 20
            ),
            candidate(
              id: "router-q8",
              bits: 8,
              bytes: 200,
              relativeMSE: 0.2,
              decodeNS: 20,
              prefillNS: 50
            ),
          ]
        ),
      ]
    )
  }

  private func candidate(
    id: String,
    calibration: ScalePlanCalibration = .standard,
    bits: Int,
    bytes: Int64,
    relativeMSE: Double,
    decodeNS: Int64,
    prefillNS: Int64
  ) -> ScalePlanCandidate {
    ScalePlanCandidate(
      id: id,
      calibration: calibration,
      bits: bits,
      payloadBits: bytes * 8,
      metadataBytes: 0,
      reconstructionMSE: relativeMSE,
      relativeMSE: relativeMSE,
      decodeNS: decodeNS,
      verifyNS: decodeNS * 4,
      prefillNS: prefillNS,
      peakBytes: bytes,
      contiguityCopyBytes: 0,
      coldCompileMS: 0
    )
  }
}

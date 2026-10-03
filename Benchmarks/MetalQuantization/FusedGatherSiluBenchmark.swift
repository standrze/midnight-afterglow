#if os(macOS)
import Foundation
import MLX
import MLXLMCommon

@_spi(Benchmark) import WickModelSupport

private struct FusedGatherTiming: Codable {
  let order: String
  let baselineMilliseconds: Double
  let candidateMilliseconds: Double
  let baselineTotalMilliseconds: Double
  let candidateTotalMilliseconds: Double
}

private struct FusedGatherError: Codable {
  let caseIndex: Int
  let experts: [Int]
  let maxAbsoluteVersusBaseline: Float
  let rmsVersusBaseline: Float
  let projectionMaxAbsoluteVersusBaseline: Float
  let projectionsMatchExactly: Bool
  let maxAbsoluteVersusFP32Oracle: Float
  let baselineMaxAbsoluteVersusFP32Oracle: Float
  let allCloseToBaseline: Bool
  let baselineHash: String
  let candidateHash: String
}

private struct FusedGatherReport: Codable {
  let format: Int
  let createdAt: String
  let status: String
  let hardware: String
  let shape: String
  let dtype: String
  let scope: String
  let firstCallOrder: String
  let firstCallBaselineMilliseconds: Double
  let firstCallCandidateMilliseconds: Double
  let synchronizedTrials: [FusedGatherTiming]
  let queuedTrials: [FusedGatherTiming]
  let queueDepth: Int
  let warmups: Int
  let errors: [FusedGatherError]
  let baselineSynchronizedMedianMilliseconds: Double
  let candidateSynchronizedMedianMilliseconds: Double
  let baselineQueuedMedianMilliseconds: Double
  let candidateQueuedMedianMilliseconds: Double
  let queuedExecutionSpeedup: Double
  let queuedTotalSpeedup: Double
}

func benchmarkFusedGatherSilu(warmups: Int, iterations: Int, queueDepth: Int, queueRounds: Int) {
  Device.withDefaultDevice(.gpu) {
    Memory.cacheLimit = 256 * 1_024 * 1_024
    let count = max(iterations, queueDepth)
    let weights = MLXRandom.normal([256, 1024, 2048], dtype: .bfloat16,
      scale: 0.02, key: MLXRandom.key(71))
    let quant = quantized(weights, groupSize: 64, bits: 4)
    guard let biases = quant.biases else { fatalError("Affine Q4 requires biases") }
    let inputs = (0..<count).map { index in
      MLXRandom.normal([1, 1, 1, 1, 2048], dtype: .bfloat16,
        key: MLXRandom.key(UInt64(index + 101)))
    }
    // Include edge expert IDs and non-sorted order. The duplicate case checks
    // that slot output indexing is independent of the expert ID.
    let selections = (0..<count).map { index -> [Int] in
      if index == 0 { return [255, 0, 3, 71, 128, 14, 219, 42] }
      if index == 1 { return [7, 7, 0, 255, 7, 128, 42, 42] }
      return (0..<8).map { (index * 17 + $0 * 29) % 256 }
    }
    let indices = selections.map { MLXArray($0, [1, 1, 8]).asType(.uint32) }
    eval(inputs + indices + [quant.wq, quant.scales, biases])
    Stream.defaultStream(.gpu).synchronize()

    // Capture is a separate diagnostic kernel, never used in timed sections.
    let diagnosticKernel = MLXFast.metalKernel(name: "benchmark_affine_q4_gather_silu_capture_v2",
      inputNames: ["x", "packed", "scales", "biases", "indices"],
      outputNames: ["output", "projection"],
      source: LagunaFusedGatherSiluKernel.source.replacingOccurrences(of: "// PROJECTION_CAPTURE_POINT", with:
        "projection[slot * 1024 + first_row + r] = gate; projection[slot * 1024 + first_row + r + 512] = up;"))
    func baselineProjection(_ index: Int) -> MLXArray {
      let i = index % count
      return gatherQuantizedMM(inputs[i], quant.wq, scales: quant.scales,
        biases: biases, rhsIndices: indices[i], transpose: true, groupSize: 64, bits: 4)
    }
    func baseline(_ index: Int) -> MLXArray {
      let split = MLX.split(baselineProjection(index), parts: 2, axis: -1)
      return compiledSiluProduct(split[0], split[1])
    }
    func candidate(_ index: Int) -> MLXArray {
      let i = index % count
      guard let output = LagunaFusedGatherSiluKernel.callAsFunction(
        input: inputs[i], weight: quant.wq, scales: quant.scales,
        biases: biases, indices: indices[i])
      else { fatalError("Fused gather fixture unexpectedly failed eligibility") }
      return output
    }
    func measure(_ op: (Int) -> MLXArray, start: Int, depth: Int) -> (Double, Double) {
      let totalStart = ContinuousClock.now
      // Retain and evaluate every output. Independent inputs prevent graph CSE
      // from making queued repetitions disappear.
      let outputs = (0..<depth).map { op(start + $0) }
      let executeStart = ContinuousClock.now
      eval(outputs)
      Stream.defaultStream(.gpu).synchronize()
      let end = ContinuousClock.now
      return (fgMilliseconds(executeStart.duration(to: end)) / Double(depth),
        fgMilliseconds(totalStart.duration(to: end)) / Double(depth))
    }
    let candidateFirst = CommandLine.arguments.contains("--fused-gather-silu-candidate-first")
    let coldA: (Double, Double)
    let coldB: (Double, Double)
    if candidateFirst {
      coldB = measure(candidate, start: 0, depth: 1)
      coldA = measure(baseline, start: 0, depth: 1)
    } else {
      coldA = measure(baseline, start: 0, depth: 1)
      coldB = measure(candidate, start: 0, depth: 1)
    }

    var errors: [FusedGatherError] = []
    // The FP32 oracle uses only the selected expert weights, avoiding a full
    // dequantized 256-expert copy. It is evaluated outside all timed sections.
    for i in 0..<min(count, 8) {
      let selected = indices[i].flattened()
      let decoded = dequantized(quant.wq[selected], scales: quant.scales[selected],
        biases: biases[selected], groupSize: 64, bits: 4).asType(.float32)
      let projected = matmul(inputs[i].reshaped([1, 2048]).asType(.float32),
        decoded.swappedAxes(-1, -2))
      let parts = MLX.split(projected, parts: 2, axis: -1)
      let oracle = (parts[0] * sigmoid(parts[0]) * parts[1]).reshaped([1, 1, 8, 1, 512])
      let a = baseline(i)
      let b = candidate(i)
      let expectedProjection = baselineProjection(i)
      let captured = diagnosticKernel([inputs[i], quant.wq, quant.scales, biases, indices[i]],
        template: [("T", DType.bfloat16)], grid: (128 * 64, 8, 1), threadGroup: (64, 1, 1),
        outputShapes: [[1, 1, 8, 1, 512], [1, 1, 8, 1, 1024]],
        outputDTypes: [.bfloat16, .bfloat16])
      eval([a, b, oracle, expectedProjection] + captured)
      let projectionError = abs(expectedProjection.asType(.float32)
        - captured[1].asType(.float32)).max().item(Float.self)
      let delta = a.asType(.float32) - b.asType(.float32)
      let close = allClose(a, b, rtol: 0.01, atol: 0.02).item(Bool.self)
      errors.append(FusedGatherError(caseIndex: i, experts: selections[i],
        maxAbsoluteVersusBaseline: abs(delta).max().item(Float.self),
        rmsVersusBaseline: sqrt(mean(delta * delta)).item(Float.self),
        projectionMaxAbsoluteVersusBaseline: projectionError,
        projectionsMatchExactly: projectionError == 0,
        maxAbsoluteVersusFP32Oracle: abs(b.asType(.float32) - oracle).max().item(Float.self),
        baselineMaxAbsoluteVersusFP32Oracle: abs(a.asType(.float32) - oracle).max().item(Float.self),
        allCloseToBaseline: close, baselineHash: fgHash(a), candidateHash: fgHash(b)))
    }
    let passed = errors.allSatisfy { $0.allCloseToBaseline && $0.maxAbsoluteVersusFP32Oracle.isFinite }
    var synchronized: [FusedGatherTiming] = []
    var queued: [FusedGatherTiming] = []
    if passed {
      for i in 0..<warmups {
        if i.isMultiple(of: 2) { eval(baseline(i)); eval(candidate(i)) }
        else { eval(candidate(i)); eval(baseline(i)) }
      }
      Stream.defaultStream(.gpu).synchronize()
      func pair(_ round: Int, depth: Int) -> FusedGatherTiming {
        let a: (Double, Double)
        let b: (Double, Double)
        let ab = round.isMultiple(of: 2) != candidateFirst
        if ab {
          a = measure(baseline, start: round * depth, depth: depth)
          b = measure(candidate, start: round * depth, depth: depth)
        } else {
          b = measure(candidate, start: round * depth, depth: depth)
          a = measure(baseline, start: round * depth, depth: depth)
        }
        return FusedGatherTiming(order: ab ? "AB" : "BA", baselineMilliseconds: a.0,
          candidateMilliseconds: b.0, baselineTotalMilliseconds: a.1, candidateTotalMilliseconds: b.1)
      }
      synchronized = (0..<iterations).map { pair($0, depth: 1) }
      queued = (0..<queueRounds).map { pair($0, depth: queueDepth) }
    }
    let aQueued = fgMedian(queued.map(\.baselineMilliseconds))
    let bQueued = fgMedian(queued.map(\.candidateMilliseconds))
    let aTotal = fgMedian(queued.map(\.baselineTotalMilliseconds))
    let bTotal = fgMedian(queued.map(\.candidateTotalMilliseconds))
    let report = FusedGatherReport(format: 2, createdAt: ISO8601DateFormatter().string(from: Date()),
      status: passed ? "measured_synthetic_prototype" : "numerical_gate_failed",
      hardware: GPU.deviceInfo().architecture, shape: "E256 top8 K2048 gateUp1024 hidden512 Q4/G64",
      dtype: "bfloat16", scope: "Synthetic primitive timing only; shared with the opt-in runner experiment; no model-speed claim. First calls include JIT and are order-sensitive. Numerical gate rtol=0.01 atol=0.02; exact hashes reported separately.",
      firstCallOrder: candidateFirst ? "BA" : "AB", firstCallBaselineMilliseconds: coldA.1,
      firstCallCandidateMilliseconds: coldB.1, synchronizedTrials: synchronized, queuedTrials: queued,
      queueDepth: queueDepth, warmups: warmups, errors: errors,
      baselineSynchronizedMedianMilliseconds: fgMedian(synchronized.map(\.baselineMilliseconds)),
      candidateSynchronizedMedianMilliseconds: fgMedian(synchronized.map(\.candidateMilliseconds)),
      baselineQueuedMedianMilliseconds: aQueued, candidateQueuedMedianMilliseconds: bQueued,
      queuedExecutionSpeedup: bQueued > 0 ? aQueued / bQueued : 0,
      queuedTotalSpeedup: bTotal > 0 ? aTotal / bTotal : 0)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.nonConformingFloatEncodingStrategy = .convertToString(
      positiveInfinity: "Infinity", negativeInfinity: "-Infinity", nan: "NaN")
    do {
      let json = try encoder.encode(report)
      if let flag = CommandLine.arguments.firstIndex(of: "--fused-gather-silu-output") {
        guard flag + 1 < CommandLine.arguments.count else { fatalError("Missing output path") }
        try json.write(to: URL(fileURLWithPath: CommandLine.arguments[flag + 1]), options: .atomic)
      }
      print(String(decoding: json, as: UTF8.self))
    } catch { fatalError("Could not write fused gather benchmark report: \(error)") }
    if !passed { exit(1) }
  }
}

private func fgMilliseconds(_ duration: Duration) -> Double {
  let c = duration.components
  return Double(c.seconds) * 1_000 + Double(c.attoseconds) / 1_000_000_000_000_000
}

private func fgMedian(_ values: [Double]) -> Double {
  guard !values.isEmpty else { return 0 }
  let sorted = values.sorted()
  let mid = sorted.count / 2
  return sorted.count.isMultiple(of: 2) ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
}

private func fgHash(_ array: MLXArray) -> String {
  var hash: UInt64 = 14_695_981_039_346_656_037
  for value in array.asType(.float32).asArray(Float.self) {
    var bits = value.bitPattern
    for _ in 0..<4 {
      hash ^= UInt64(bits & 255)
      hash &*= 1_099_511_628_211
      bits >>= 8
    }
  }
  return String(format: "%016llx", hash)
}

#else
func benchmarkFusedGatherSilu(warmups: Int, iterations: Int, queueDepth: Int, queueRounds: Int) {
  fatalError("--fused-gather-silu-ab requires macOS and Metal")
}
#endif

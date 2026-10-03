import Foundation
import MLX
import MLXLMCommon

private struct QuantizationCase {
  let name: String
  let mode: QuantizationMode
  let groupSize: Int
  let bits: Int
}

private struct Measurement {
  let workload: String
  let quantization: String
  let synchronizedMedianMilliseconds: Double
  let synchronizedMinimumMilliseconds: Double
  let queuedMilliseconds: Double
}

private struct PairedMeasurement {
  let workload: String
  let firstName: String
  let secondName: String
  let firstTotalMilliseconds: [Double]
  let secondTotalMilliseconds: [Double]
  let firstExecutionMilliseconds: [Double]
  let secondExecutionMilliseconds: [Double]
}

private struct QMVValidation {
  let maxAbsoluteError: Double
  let outputHash: String
}

// Validate before constructing MLX graphs: asking for usage must never start a GPU run.
validateCommandLine()

private let quantizationCases = [
  QuantizationCase(name: "affine-4bit-g64", mode: .affine, groupSize: 64, bits: 4),
  QuantizationCase(name: "affine-4bit-g128", mode: .affine, groupSize: 128, bits: 4),
  QuantizationCase(name: "affine-8bit-g64", mode: .affine, groupSize: 64, bits: 8),
  QuantizationCase(name: "affine-8bit-g128", mode: .affine, groupSize: 128, bits: 8),
  QuantizationCase(name: "mxfp4-g32", mode: .mxfp4, groupSize: 32, bits: 4),
  QuantizationCase(name: "nvfp4-g16", mode: .nvfp4, groupSize: 16, bits: 4),
]

private let warmupCount = integerArgument("--warmup", defaultValue: 8)
private let iterationCount = integerArgument("--iterations", defaultValue: 40)
private let queueDepth = integerArgument("--queue-depth", defaultValue: 32)
private let queueRounds = integerArgument("--queue-rounds", defaultValue: 7)
private let scaleSearchOnly = CommandLine.arguments.contains("--scale-search-only")
private let formatABOnly = CommandLine.arguments.contains("--format-ab")
private let lagunaGraphABOnly = CommandLine.arguments.contains("--laguna-graph-ab")
private let mistralGraphABOnly = CommandLine.arguments.contains("--mistral-graph-ab")
private let qmvSpecializationOnly = CommandLine.arguments.contains("--qmv-specialization")
private let routerPrecisionABOnly = CommandLine.arguments.contains("--router-precision-ab")
private let fusedGatherSiluABOnly = CommandLine.arguments.contains("--fused-gather-silu-ab")
private let gptOSS20BOnly = CommandLine.arguments.contains("--gpt-oss-20b")

private let siluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
  shapeless: true
) { gate, up in
  gate * sigmoid(gate) * up
}

guard warmupCount >= 1, iterationCount >= 3, queueDepth >= 2, queueRounds >= 3 else {
  fatalError(
    "--warmup must be at least 1, --iterations and --queue-rounds at least 3, and --queue-depth at least 2"
  )
}

if gptOSS20BOnly {
  benchmarkGPTOSS20B()
} else if fusedGatherSiluABOnly {
  benchmarkFusedGatherSilu(warmups: warmupCount, iterations: iterationCount,
    queueDepth: queueDepth, queueRounds: queueRounds)
} else if scaleSearchOnly {
  benchmarkAffineScaleSearch()
} else if formatABOnly {
  benchmarkFormatAB()
} else if lagunaGraphABOnly {
  benchmarkLagunaGraphAB()
} else if mistralGraphABOnly {
  benchmarkMistralGraphAB()
} else if qmvSpecializationOnly {
  benchmarkAffineQ4QMVSpecialization()
} else if routerPrecisionABOnly {
  benchmarkRouterPrecisionAB()
} else {
  MLXRandom.seed(7)
  Memory.cacheLimit = 256 * 1_024 * 1_024

  print("Laguna-shaped MLX Q4R8 quantization benchmark")
  print(
    "warmup=\(warmupCount) iterations=\(iterationCount) "
      + "queue_depth=\(queueDepth) queue_rounds=\(queueRounds) dtype=bfloat16"
  )
  print("workloads: dense 2048->6144; MoE gate/up 8x(2048->1024); MoE down 8x(512->2048)")

  // Pay one-time command-queue, shader, and frequency-ramp costs before the first
  // reported case. Otherwise affine Q4 is unfairly charged for process startup.
  _ = benchmarkDense(
    quantizationCases[0],
    inputDimensions: 2_048,
    outputDimensions: 6_144
  )
  Memory.clearCache()

  var measurements: [Measurement] = []
  for quantization in quantizationCases {
    measurements.append(
      benchmarkDense(
        quantization,
        inputDimensions: 2_048,
        outputDimensions: 6_144
      )
    )
    measurements.append(
      benchmarkExpertGather(
        quantization,
        workload: "moe-gate-up",
        inputDimensions: 2_048,
        outputDimensions: 1_024,
        numberOfExperts: 256,
        selectedExperts: 8
      )
    )
    measurements.append(
      benchmarkExpertGather(
        quantization,
        workload: "moe-down",
        inputDimensions: 512,
        outputDimensions: 2_048,
        numberOfExperts: 256,
        selectedExperts: 8
      )
    )
    Memory.clearCache()
  }

  print(
    "\nworkload\tquantization\tsync_median_ms\tsync_min_ms\tqueued_ms_per_op"
      + "\tqueued_relative_to_affine4_g64"
  )
  for measurement in measurements {
    let baseline = measurements.first {
      $0.workload == measurement.workload && $0.quantization == "affine-4bit-g64"
    }!
    let relative = baseline.queuedMilliseconds / measurement.queuedMilliseconds
    print(
      "\(measurement.workload)\t\(measurement.quantization)\t"
        + "\(format(measurement.synchronizedMedianMilliseconds))\t"
        + "\(format(measurement.synchronizedMinimumMilliseconds))\t"
        + "\(format(measurement.queuedMilliseconds))\t"
        + "\(format(relative))x"
    )
  }

  let sharedGateUp = benchmarkSharedGateUpFusion(quantizationCases[0])
  print("")
  print("shared_gate_up_layout\tsync_median_ms\tsync_min_ms\tqueued_ms_per_op")
  print(
    "separate\t\(format(median(sharedGateUp.separate.synchronized)))\t"
      + "\(format(sharedGateUp.separate.synchronized.min()!))\t"
      + "\(format(median(sharedGateUp.separate.queued)))"
  )
  print(
    "fused\t\(format(median(sharedGateUp.fused.synchronized)))\t"
      + "\(format(sharedGateUp.fused.synchronized.min()!))\t"
      + "\(format(median(sharedGateUp.fused.queued)))"
  )
  print(
    "fused_speedup\t"
      + "\(format(median(sharedGateUp.separate.queued) / median(sharedGateUp.fused.queued)))x"
  )
}

private func benchmarkGPTOSS20B() {
  MLXRandom.seed(7)
  Memory.cacheLimit = 256 * 1_024 * 1_024
  let affine = QuantizationCase(name: "affine-4bit-g64", mode: .affine, groupSize: 64, bits: 4)
  let affine128 = QuantizationCase(name: "affine-4bit-g128", mode: .affine, groupSize: 128, bits: 4)
  let mxfp4 = QuantizationCase(name: "mxfp4-g32", mode: .mxfp4, groupSize: 32, bits: 4)

  // Geometry: https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json
  print("GPT-OSS-20B-shaped synthetic Metal format A/B")
  print("SYNTHETIC: random BF16 source weights quantized independently; not checkpoint quality, model tokens/s, or a native MXFP4 conversion test.")
  print("E32 top4 hidden=2880 intermediate=2880 Q=4096 KV=512 dtype=bfloat16")
  print("Separate expert gate/up, learned biases, GPT-OSS clamp and alpha=1.702; down includes bias and weighted reduction. Routing, attention and KV cache are excluded.")
  print("warmup=\(warmupCount) iterations=\(iterationCount) queue_depth=\(queueDepth) queue_rounds=\(max(queueRounds, 5))")
  print("SKIP affine-4bit-g128 for K=2880: input width is not divisible by 128; no padding or changed model geometry. G128 is valid for O projection K=4096.")

  // Pay the initial shader/queue costs before retaining a paired result.
  _ = benchmarkDense(affine, inputDimensions: 2_880, outputDimensions: 4_096)
  Memory.clearCache()

  var measurements: [PairedMeasurement] = []
  measurements.append(benchmarkDensePair(affine, mxfp4, workload: "gptoss-query-qmv",
    inputDimensions: 2_880, outputDimensions: 4_096))
  Memory.clearCache()
  measurements.append(benchmarkDensePair(affine, mxfp4, workload: "gptoss-output-qmv",
    inputDimensions: 4_096, outputDimensions: 2_880))
  Memory.clearCache()
  measurements.append(benchmarkDensePair(affine, affine128, workload: "gptoss-output-qmv",
    inputDimensions: 4_096, outputDimensions: 2_880))
  Memory.clearCache()
  measurements.append(benchmarkGPTOSSExpertPair(affine, mxfp4, gateUp: true))
  Memory.clearCache()
  measurements.append(benchmarkGPTOSSExpertPair(affine, mxfp4, gateUp: false))
  Memory.clearCache()

  print("workload\tfirst\tsecond\tfirst_total_ms\tsecond_total_ms\tsecond_total_speedup\tfirst_execution_ms\tsecond_execution_ms\tsecond_execution_speedup")
  for measurement in measurements {
    let firstTotal = median(measurement.firstTotalMilliseconds)
    let secondTotal = median(measurement.secondTotalMilliseconds)
    let firstExecution = median(measurement.firstExecutionMilliseconds)
    let secondExecution = median(measurement.secondExecutionMilliseconds)
    print("\(measurement.workload)\t\(measurement.firstName)\t\(measurement.secondName)\t"
      + "\(formatPrecise(firstTotal))\t\(formatPrecise(secondTotal))\t\(formatPrecise(firstTotal / secondTotal))x\t"
      + "\(formatPrecise(firstExecution))\t\(formatPrecise(secondExecution))\t\(formatPrecise(firstExecution / secondExecution))x")
    // Retain individual rounds so order effects and noise remain inspectable.
    print("rounds \(measurement.workload) \(measurement.firstName) total_ms=\(measurement.firstTotalMilliseconds) execution_ms=\(measurement.firstExecutionMilliseconds)")
    print("rounds \(measurement.workload) \(measurement.secondName) total_ms=\(measurement.secondTotalMilliseconds) execution_ms=\(measurement.secondExecutionMilliseconds)")
  }
}

private struct GPTOSSBenchmarkProjection {
  let quantization: QuantizationCase
  let weight: MLXArray
  let scales: MLXArray
  let affineBiases: MLXArray?
  let learnedBias: MLXArray

  func callAsFunction(_ input: MLXArray, _ indices: MLXArray) -> MLXArray {
    let output = gatherQuantizedMM(input, weight, scales: scales, biases: affineBiases,
      rhsIndices: indices, transpose: true, groupSize: quantization.groupSize,
      bits: quantization.bits, mode: quantization.mode)
    return output + expandedDimensions(learnedBias[indices], axis: -2)
  }
}

private func makeGPTOSSProjectionPair(_ first: QuantizationCase, _ second: QuantizationCase)
  -> (GPTOSSBenchmarkProjection, GPTOSSBenchmarkProjection) {
  // One source projection is materialized at a time (32 * 2880 * 2880 BF16 values).
  // Realize both packed forms before returning so the source can be released.
  let source = MLXRandom.normal([32, 2_880, 2_880], dtype: .bfloat16, scale: 0.02)
  let learnedBias = MLXRandom.normal([32, 2_880], dtype: .bfloat16, scale: 0.02)
  func pack(_ quantization: QuantizationCase) -> GPTOSSBenchmarkProjection {
    let arrays = quantized(source, groupSize: quantization.groupSize,
      bits: quantization.bits, mode: quantization.mode)
    evaluate(arrays.wq, arrays.scales, arrays.biases, learnedBias)
    return GPTOSSBenchmarkProjection(quantization: quantization, weight: arrays.wq,
      scales: arrays.scales, affineBiases: arrays.biases, learnedBias: learnedBias)
  }
  return (pack(first), pack(second))
}

private func benchmarkGPTOSSExpertPair(
  _ first: QuantizationCase, _ second: QuantizationCase, gateUp: Bool
) -> PairedMeasurement {
  let inputs = makeInputs(gateUp ? [1, 1, 1, 1, 2_880] : [1, 1, 4, 1, 2_880])
  let indices = makeExpertIndices(numberOfExperts: 32, selectedExperts: 4)
  let (firstProjection, secondProjection) = makeGPTOSSProjectionPair(first, second)
  let upPair = gateUp ? makeGPTOSSProjectionPair(first, second) : nil
  let expertWeights = softmax(MLXRandom.normal([1, 1, 4], dtype: .bfloat16), axis: -1, precise: true)
  let activation: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { up, gate in
    let clippedGate = clip(gate, max: MLXArray(7.0))
    let clippedUp = clip(up, min: MLXArray(-7.0), max: MLXArray(7.0))
    return (clippedGate * sigmoid(Float(1.702) * clippedGate)) * (clippedUp + 1)
  }
  eval(inputs)
  eval(indices)
  eval(expertWeights)
  Stream.gpu.synchronize()

  func output(_ projection: GPTOSSBenchmarkProjection, _ up: GPTOSSBenchmarkProjection?, _ index: Int) -> [MLXArray] {
    let input = inputs[index % inputs.count]
    let selected = indices[index % indices.count]
    let projected = projection(input, selected)
    if let up { return [activation(up(input, selected), projected)] }
    return [(projected.squeezed(axis: -2) * expandedDimensions(expertWeights, axis: -1)).sum(axis: -2)]
  }
  let expectedShape = gateUp ? [1, 1, 4, 1, 2_880] : [1, 1, 2_880]
  for sample in [output(firstProjection, upPair?.0, 0)[0], output(secondProjection, upPair?.1, 0)[0]] {
    precondition(sample.shape == expectedShape && all(isFinite(sample)).item(Bool.self),
      "GPT-OSS synthetic workload must produce finite outputs with its native shape")
  }
  return benchmarkPair(workload: gateUp ? "gptoss-separate-gate-up-bias-clamp" : "gptoss-down-bias-weighted-reduction",
    firstName: first.name, secondName: second.name,
    first: { output(firstProjection, upPair?.0, $0) },
    second: { output(secondProjection, upPair?.1, $0) })
}

private func benchmarkRouterPrecisionAB() {
  MLXRandom.seed(7)
  Memory.cacheLimit = 256 * 1_024 * 1_024

  let q4 = QuantizationCase(
    name: "affine-4bit-g64", mode: .affine, groupSize: 64, bits: 4)
  let q8 = QuantizationCase(
    name: "affine-8bit-g64", mode: .affine, groupSize: 64, bits: 8)
  let measurement = benchmarkDensePair(
    q4, q8, inputDimensions: 2_048, outputDimensions: 256)
  let q4Total = median(measurement.firstTotalMilliseconds)
  let q8Total = median(measurement.secondTotalMilliseconds)
  let q4Execution = median(measurement.firstExecutionMilliseconds)
  let q8Execution = median(measurement.secondExecutionMilliseconds)

  print("Laguna 2048->256 router precision Metal A/B")
  print(
    "warmup=\(warmupCount) queue_depth=\(queueDepth) "
      + "queue_rounds=\(max(queueRounds, 5)) dtype=bfloat16"
  )
  print(
    "workload\tq4_total_ms\tq8_total_ms\tq4_total_speedup\t"
      + "q4_execution_ms\tq8_execution_ms\tq4_execution_speedup")
  print(
    "router-qmv\t\(formatPrecise(q4Total))\t\(formatPrecise(q8Total))\t"
      + "\(formatPrecise(q8Total / q4Total))x\t\(formatPrecise(q4Execution))\t"
      + "\(formatPrecise(q8Execution))\t"
      + "\(formatPrecise(q8Execution / q4Execution))x")
}

private func benchmarkAffineQ4QMVSpecialization() {
  MLXRandom.seed(7)
  Memory.cacheLimit = 256 * 1_024 * 1_024

  let quantization = QuantizationCase(
    name: "affine-4bit-g64", mode: .affine, groupSize: 64, bits: 4)
  let requestedResults = ProcessInfo.processInfo.environment[
    "MLX_METAL_AFFINE_QMV_RESULTS_PER_SIMDGROUP"
  ] ?? "stock"
  let minimumOutputs = ProcessInfo.processInfo.environment[
    "MLX_METAL_AFFINE_QMV_MIN_OUTPUTS"
  ] ?? "0"

  print("Laguna affine-Q4/G64 QMV specialization benchmark")
  print(
    "results_per_simdgroup=\(requestedResults) minimum_outputs=\(minimumOutputs) "
      + "warmup=\(warmupCount) "
      + "iterations=\(iterationCount) queue_depth=\(queueDepth) "
      + "queue_rounds=\(queueRounds) dtype=bfloat16"
  )
  print(
    "workload\tmax_abs_error\toutput_hash\tsync_median_ms\tsync_min_ms"
      + "\tqueued_ms_per_op"
  )

  let denseShapes = [
    (workload: "dense-2048x6144", input: 2_048, output: 6_144),
    (workload: "dense-2048x8192", input: 2_048, output: 8_192),
    (workload: "dense-2048x16384", input: 2_048, output: 16_384),
    (workload: "dense-lm-head-2048x100352", input: 2_048, output: 100_352),
  ]
  for shape in denseShapes {
    let error = validateDenseAffineQ4(
      inputDimensions: shape.input, outputDimensions: shape.output)
    let measurement = benchmarkDense(
      quantization,
      workload: shape.workload,
      inputDimensions: shape.input,
      outputDimensions: shape.output
    )
    printSpecializationMeasurement(measurement, validation: error)
    Memory.clearCache()
  }

  let gatherShapes = [
    (workload: "gather-gate-up-2048x1024", input: 2_048, output: 1_024),
    (workload: "gather-down-512x2048", input: 512, output: 2_048),
  ]
  for shape in gatherShapes {
    let error = validateGatherAffineQ4(
      inputDimensions: shape.input, outputDimensions: shape.output)
    let measurement = benchmarkExpertGather(
      quantization,
      workload: shape.workload,
      inputDimensions: shape.input,
      outputDimensions: shape.output,
      numberOfExperts: 256,
      selectedExperts: 8
    )
    printSpecializationMeasurement(measurement, validation: error)
    Memory.clearCache()
  }
}

private func printSpecializationMeasurement(
  _ measurement: Measurement,
  validation: QMVValidation
) {
  print(
    "\(measurement.workload)\t\(format(validation.maxAbsoluteError))\t"
      + "\(validation.outputHash)\t"
      + "\(format(measurement.synchronizedMedianMilliseconds))\t"
      + "\(format(measurement.synchronizedMinimumMilliseconds))\t"
      + "\(format(measurement.queuedMilliseconds))"
  )
}

private func validateDenseAffineQ4(
  inputDimensions: Int,
  outputDimensions: Int
) -> QMVValidation {
  let input = MLXRandom.normal([1, inputDimensions], dtype: .bfloat16)
  let weight = MLXRandom.normal(
    [outputDimensions, inputDimensions], dtype: .bfloat16, scale: 0.02)
  let packed = quantized(weight, groupSize: 64, bits: 4, mode: .affine)
  let restored = dequantized(
    packed.wq,
    scales: packed.scales,
    biases: packed.biases,
    groupSize: 64,
    bits: 4,
    mode: .affine,
    dtype: .bfloat16
  )
  let expected = matmul(input, restored.T)
  let actual = quantizedMM(
    input,
    packed.wq,
    scales: packed.scales,
    biases: packed.biases,
    transpose: true,
    groupSize: 64,
    bits: 4,
    mode: .affine
  )
  return validateQMVOutput(actual: actual, expected: expected)
}

private func validateGatherAffineQ4(
  inputDimensions: Int,
  outputDimensions: Int
) -> QMVValidation {
  let numberOfExperts = 8
  let selectedExperts = 8
  let input = MLXRandom.normal(
    [1, 1, 1, 1, inputDimensions], dtype: .bfloat16)
  let weight = MLXRandom.normal(
    [numberOfExperts, outputDimensions, inputDimensions],
    dtype: .bfloat16,
    scale: 0.02
  )
  let indices = MLXArray(Array(0..<selectedExperts), [1, 1, selectedExperts])
  let packed = quantized(weight, groupSize: 64, bits: 4, mode: .affine)
  let restored = dequantized(
    packed.wq,
    scales: packed.scales,
    biases: packed.biases,
    groupSize: 64,
    bits: 4,
    mode: .affine,
    dtype: .bfloat16
  )
  let expected = gatherMM(
    input, restored.swappedAxes(-1, -2), rhsIndices: indices)
  let actual = gatherQuantizedMM(
    input,
    packed.wq,
    scales: packed.scales,
    biases: packed.biases,
    rhsIndices: indices,
    transpose: true,
    groupSize: 64,
    bits: 4,
    mode: .affine
  )
  return validateQMVOutput(actual: actual, expected: expected)
}

private func validateQMVOutput(actual: MLXArray, expected: MLXArray) -> QMVValidation {
  precondition(actual.shape == expected.shape, "specialized QMV changed the output shape")
  let actualFloat = actual.asType(.float32)
  let close = actual.allClose(expected, rtol: 0.02, atol: 0.04)
  let maxError = MLX.max((actualFloat - expected.asType(.float32)).abs())
  evaluate(actualFloat, close, maxError)
  let isClose = close.item(Bool.self)
  let maxErrorValue = Double(maxError.item(Float.self))
  if !isClose {
    print(
      "validation_failed shape=\(actual.shape) max_abs_error=\(maxErrorValue) "
        + "rtol=0.02 atol=0.04"
    )
  }
  precondition(isClose, "specialized QMV failed numerical validation")

  var outputHash: UInt64 = 1_469_598_103_934_665_603
  for value in actualFloat.asArray(Float.self) {
    outputHash ^= UInt64(value.bitPattern)
    outputHash &*= 1_099_511_628_211
  }
  return QMVValidation(
    maxAbsoluteError: maxErrorValue,
    outputHash: String(format: "%016llx", outputHash)
  )
}

private func benchmarkFormatAB() {
  MLXRandom.seed(7)
  Memory.cacheLimit = 256 * 1_024 * 1_024

  let affine = QuantizationCase(
    name: "q4r8-affine-4bit-g64", mode: .affine, groupSize: 64, bits: 4)
  let nvfp4 = QuantizationCase(
    name: "nvfp4-g16", mode: .nvfp4, groupSize: 16, bits: 4)

  print("Laguna-shaped interleaved Metal format A/B")
  print(
    "warmup=\(warmupCount) iterations=\(iterationCount) "
      + "queue_depth=\(queueDepth) queue_rounds=\(queueRounds) dtype=bfloat16"
  )

  let measurements = [
    benchmarkDensePair(
      affine, nvfp4, inputDimensions: 2_048, outputDimensions: 6_144),
    benchmarkExpertPair(
      affine, nvfp4, workload: "moe-gate-up", inputDimensions: 2_048,
      outputDimensions: 1_024, numberOfExperts: 256, selectedExperts: 8),
    benchmarkExpertPair(
      affine, nvfp4, workload: "moe-down", inputDimensions: 512,
      outputDimensions: 2_048, numberOfExperts: 256, selectedExperts: 8),
  ]

  print(
    "workload\tfirst\tsecond\tfirst_total_ms\tsecond_total_ms\t"
      + "second_total_speedup\tfirst_execution_ms\tsecond_execution_ms\t"
      + "second_execution_speedup"
  )
  for measurement in measurements {
    let firstTotal = median(measurement.firstTotalMilliseconds)
    let secondTotal = median(measurement.secondTotalMilliseconds)
    let firstExecution = median(measurement.firstExecutionMilliseconds)
    let secondExecution = median(measurement.secondExecutionMilliseconds)
    print(
      "\(measurement.workload)\t\(measurement.firstName)\t\(measurement.secondName)\t"
        + "\(format(firstTotal))\t\(format(secondTotal))\t"
        + "\(format(firstTotal / secondTotal))x\t"
        + "\(format(firstExecution))\t\(format(secondExecution))\t"
        + "\(format(firstExecution / secondExecution))x"
    )
  }
}

private func benchmarkDensePair(
  _ first: QuantizationCase,
  _ second: QuantizationCase,
  workload: String = "dense-qmv",
  inputDimensions: Int,
  outputDimensions: Int
) -> PairedMeasurement {
  let inputs = makeInputs([1, inputDimensions])
  let weight = MLXRandom.normal(
    [outputDimensions, inputDimensions], dtype: .bfloat16, scale: 0.02)
  let firstPacked = quantized(
    weight, groupSize: first.groupSize, bits: first.bits, mode: first.mode)
  let secondPacked = quantized(
    weight, groupSize: second.groupSize, bits: second.bits, mode: second.mode)
  evaluate(
    firstPacked.wq, firstPacked.scales, firstPacked.biases,
    secondPacked.wq, secondPacked.scales, secondPacked.biases)

  return benchmarkPair(
    workload: workload, firstName: first.name, secondName: second.name,
    first: { index in
      [
        quantizedMM(
          inputs[index % inputs.count], firstPacked.wq,
          scales: firstPacked.scales, biases: firstPacked.biases, transpose: true,
          groupSize: first.groupSize, bits: first.bits, mode: first.mode)
      ]
    },
    second: { index in
      [
        quantizedMM(
          inputs[index % inputs.count], secondPacked.wq,
          scales: secondPacked.scales, biases: secondPacked.biases, transpose: true,
          groupSize: second.groupSize, bits: second.bits, mode: second.mode)
      ]
    })
}

private func benchmarkExpertPair(
  _ first: QuantizationCase,
  _ second: QuantizationCase,
  workload: String,
  inputDimensions: Int,
  outputDimensions: Int,
  numberOfExperts: Int,
  selectedExperts: Int
) -> PairedMeasurement {
  let inputs = makeInputs([1, 1, 1, 1, inputDimensions])
  let indices = makeExpertIndices(
    numberOfExperts: numberOfExperts, selectedExperts: selectedExperts)
  let weight = MLXRandom.normal(
    [numberOfExperts, outputDimensions, inputDimensions],
    dtype: .bfloat16, scale: 0.02)
  let firstPacked = quantized(
    weight, groupSize: first.groupSize, bits: first.bits, mode: first.mode)
  let secondPacked = quantized(
    weight, groupSize: second.groupSize, bits: second.bits, mode: second.mode)
  evaluate(
    firstPacked.wq, firstPacked.scales, firstPacked.biases,
    secondPacked.wq, secondPacked.scales, secondPacked.biases)
  eval(indices)
  Stream.gpu.synchronize()

  return benchmarkPair(
    workload: workload, firstName: first.name, secondName: second.name,
    first: { index in
      [
        gatherQuantizedMM(
          inputs[index % inputs.count], firstPacked.wq,
          scales: firstPacked.scales, biases: firstPacked.biases,
          rhsIndices: indices[index % indices.count], transpose: true,
          groupSize: first.groupSize, bits: first.bits, mode: first.mode)
      ]
    },
    second: { index in
      [
        gatherQuantizedMM(
          inputs[index % inputs.count], secondPacked.wq,
          scales: secondPacked.scales, biases: secondPacked.biases,
          rhsIndices: indices[index % indices.count], transpose: true,
          groupSize: second.groupSize, bits: second.bits, mode: second.mode)
      ]
    })
}

private func benchmarkLagunaGraphAB() {
  MLXRandom.seed(7)
  Memory.cacheLimit = 256 * 1_024 * 1_024

  let attentionOutput = MLXRandom.normal([1, 1, 48, 128], dtype: .bfloat16)
  let attentionGateLogits = MLXRandom.normal([1, 1, 48], dtype: .bfloat16)

  func eagerAttentionGate(_ index: Int) -> [MLXArray] {
    let gate = logAddExp(attentionGateLogits.asType(.float32), 0).asType(
      attentionOutput.dtype)
    return [
      (attentionOutput * gate[.ellipsis, .newAxis]).reshaped(1, 1, 48 * 128)
    ]
  }
  let compiledAttentionGate: @Sendable (MLXArray, MLXArray) -> MLXArray = compile {
    output, gateLogits in
    let gate = logAddExp(gateLogits.asType(.float32), 0).asType(output.dtype)
    return output * gate[.ellipsis, .newAxis]
  }
  func fusedAttentionGate(_ index: Int) -> [MLXArray] {
    [compiledAttentionGate(attentionOutput, attentionGateLogits).reshaped(1, 1, 48 * 128)]
  }

  let eagerGated = eagerAttentionGate(0)
  let fusedGated = fusedAttentionGate(0)
  eval(eagerGated + fusedGated)
  precondition(
    allClose(eagerGated[0], fusedGated[0], rtol: 1e-5, atol: 1e-5).item(Bool.self),
    "compiled Laguna attention gate changed its output")

  let logits = MLXRandom.normal([1, 256], dtype: .bfloat16)
  let correctionBias = MLXRandom.normal([256], dtype: .float32, scale: 0.01)

  func eagerRouter(_ index: Int) -> [MLXArray] {
    let scores = sigmoid(logits.asType(.float32))
    let indices = argPartition(
      -(scores + correctionBias), kth: 7, axis: -1
    )[.ellipsis, ..<8]
    var weights = takeAlong(scores, indices, axis: -1)
    weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
    return [weights.asType(.bfloat16), indices]
  }
  let compiledRouter: @Sendable ([MLXArray]) -> [MLXArray] = compile { inputs in
    let scores = sigmoid(inputs[0].asType(.float32))
    let indices = argPartition(
      -(scores + inputs[1]), kth: 7, axis: -1
    )[.ellipsis, ..<8]
    var weights = takeAlong(scores, indices, axis: -1)
    weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
    return [weights.asType(.bfloat16), indices]
  }
  func fusedRouter(_ index: Int) -> [MLXArray] {
    compiledRouter([logits, correctionBias])
  }

  let eagerRoute = eagerRouter(0)
  let fusedRoute = fusedRouter(0)
  eval(eagerRoute + fusedRoute)
  precondition(
    arrayEqual(eagerRoute[1], fusedRoute[1]).item(Bool.self)
      && allClose(eagerRoute[0], fusedRoute[0], rtol: 1e-5, atol: 1e-5).item(Bool.self),
    "compiled Laguna router changed its output")

  let expert = MLXRandom.normal([1, 1, 8, 2_048], dtype: .bfloat16)
  let weights = MLXRandom.uniform(low: 0, high: 1, [1, 1, 8], dtype: .bfloat16)
  let normalizedWeights = weights / weights.sum(axis: -1, keepDims: true)
  let shared = MLXRandom.normal([1, 1, 2_048], dtype: .bfloat16)
  let residual = MLXRandom.normal([1, 1, 2_048], dtype: .bfloat16)
  let scale = MLXArray(Float(2.5)).asType(.bfloat16)

  func eagerReduction(_ index: Int) -> [MLXArray] {
    [weightedExpertSum(expert, normalizedWeights) * scale + shared + residual]
  }
  let compiledReduction: @Sendable ([MLXArray]) -> [MLXArray] = compile(
    shapeless: true
  ) { inputs in
    let weighted = (inputs[0] * expandedDimensions(inputs[1].asType(inputs[0].dtype), axis: -1))
      .sum(axis: -2)
    return [weighted * inputs[2].asType(weighted.dtype) + inputs[3] + inputs[4]]
  }
  func fusedReduction(_ index: Int) -> [MLXArray] {
    compiledReduction([expert, normalizedWeights, scale, shared, residual])
  }

  let eagerReduced = eagerReduction(0)
  let fusedReduced = fusedReduction(0)
  eval(eagerReduced + fusedReduced)
  precondition(
    allClose(eagerReduced[0], fusedReduced[0], rtol: 1e-5, atol: 1e-5).item(Bool.self),
    "compiled Laguna MoE reduction changed its output")

  let measurements = [
    benchmarkPair(
      workload: "per-head-attention-gate", firstName: "current-eager",
      secondName: "compiled-fused", first: eagerAttentionGate, second: fusedAttentionGate),
    benchmarkPair(
      workload: "sigmoid-topk8-router", firstName: "current-eager",
      secondName: "compiled-fused", first: eagerRouter, second: fusedRouter),
    benchmarkPair(
      workload: "moe-weighted-shared-residual", firstName: "current-split",
      secondName: "compiled-fused", first: eagerReduction, second: fusedReduction),
  ]

  print("Laguna decode-graph interleaved Metal A/B")
  print(
    "workload\tfirst\tsecond\tfirst_total_ms\tsecond_total_ms\t"
      + "second_total_speedup\tfirst_execution_ms\tsecond_execution_ms\t"
      + "second_execution_speedup"
  )
  for measurement in measurements {
    let firstTotal = median(measurement.firstTotalMilliseconds)
    let secondTotal = median(measurement.secondTotalMilliseconds)
    let firstExecution = median(measurement.firstExecutionMilliseconds)
    let secondExecution = median(measurement.secondExecutionMilliseconds)
    print(
      "\(measurement.workload)\t\(measurement.firstName)\t\(measurement.secondName)\t"
        + "\(format(firstTotal))\t\(format(secondTotal))\t"
        + "\(format(firstTotal / secondTotal))x\t"
        + "\(format(firstExecution))\t\(format(secondExecution))\t"
        + "\(format(firstExecution / secondExecution))x"
    )
  }
}

private func benchmarkMistralGraphAB() {
  MLXRandom.seed(7)
  Memory.cacheLimit = 256 * 1_024 * 1_024

  let hidden = 4_096
  let intermediate = 14_336
  let keyValue = 1_024
  let inputs = makeInputs([1, hidden])

  func affineQ4(_ weight: MLXArray) -> (
    weight: MLXArray, scales: MLXArray, biases: MLXArray
  ) {
    let packed = quantized(weight, groupSize: 64, bits: 4, mode: .affine)
    return (packed.wq, packed.scales, packed.biases!)
  }

  func qmm(
    _ input: MLXArray,
    _ packed: (weight: MLXArray, scales: MLXArray, biases: MLXArray)
  ) -> MLXArray {
    quantizedMM(
      input, packed.weight, scales: packed.scales, biases: packed.biases,
      transpose: true, groupSize: 64, bits: 4, mode: .affine)
  }

  let gate = affineQ4(
    MLXRandom.normal([intermediate, hidden], dtype: .bfloat16, scale: 0.02))
  let up = affineQ4(
    MLXRandom.normal([intermediate, hidden], dtype: .bfloat16, scale: 0.02))
  let gateUp = (
    weight: concatenated([gate.weight, up.weight], axis: 0),
    scales: concatenated([gate.scales, up.scales], axis: 0),
    biases: concatenated([gate.biases, up.biases], axis: 0)
  )

  func eagerSeparateGateUp(_ index: Int) -> [MLXArray] {
    let input = inputs[index % inputs.count]
    let gateOutput = qmm(input, gate)
    return [gateOutput * sigmoid(gateOutput) * qmm(input, up)]
  }
  func compiledSeparateGateUp(_ index: Int) -> [MLXArray] {
    let input = inputs[index % inputs.count]
    return [compiledSiluProduct(qmm(input, gate), qmm(input, up))]
  }
  func fusedGateUp(_ index: Int) -> [MLXArray] {
    let output = qmm(inputs[index % inputs.count], gateUp)
    let parts = split(output, parts: 2, axis: -1)
    return [compiledSiluProduct(parts[0], parts[1])]
  }

  let query = affineQ4(
    MLXRandom.normal([hidden, hidden], dtype: .bfloat16, scale: 0.02))
  let key = affineQ4(
    MLXRandom.normal([keyValue, hidden], dtype: .bfloat16, scale: 0.02))
  let value = affineQ4(
    MLXRandom.normal([keyValue, hidden], dtype: .bfloat16, scale: 0.02))
  let qkv = (
    weight: concatenated([query.weight, key.weight, value.weight], axis: 0),
    scales: concatenated([query.scales, key.scales, value.scales], axis: 0),
    biases: concatenated([query.biases, key.biases, value.biases], axis: 0)
  )

  func separateQKV(_ index: Int) -> [MLXArray] {
    let input = inputs[index % inputs.count]
    return [qmm(input, query), qmm(input, key), qmm(input, value)]
  }
  func fusedQKV(_ index: Int) -> [MLXArray] {
    let output = qmm(inputs[index % inputs.count], qkv)
    return [
      output[.ellipsis, ..<hidden],
      output[.ellipsis, hidden..<(hidden + keyValue)],
      output[.ellipsis, (hidden + keyValue)..<(hidden + 2 * keyValue)],
    ]
  }

  let eagerGateUpOutput = eagerSeparateGateUp(0)
  let compiledGateUpOutput = compiledSeparateGateUp(0)
  let fusedGateUpOutput = fusedGateUp(0)
  let separateQKVOutput = separateQKV(0)
  let fusedQKVOutput = fusedQKV(0)
  eval(
    eagerGateUpOutput + compiledGateUpOutput + fusedGateUpOutput
      + separateQKVOutput + fusedQKVOutput)
  precondition(
    allClose(
      eagerGateUpOutput[0], compiledGateUpOutput[0],
      rtol: 1e-5, atol: 1e-5
    ).item(Bool.self)
      && allClose(
        eagerGateUpOutput[0], fusedGateUpOutput[0],
        rtol: 1e-5, atol: 1e-5
      ).item(Bool.self),
    "fused Mistral gate/up changed its output")
  precondition(
    zip(separateQKVOutput, fusedQKVOutput).allSatisfy {
      allClose($0, $1, rtol: 1e-5, atol: 1e-5).item(Bool.self)
    },
    "fused Mistral QKV changed its output")

  let measurements = [
    benchmarkPair(
      workload: "mistral7b-swiglu", firstName: "current-eager",
      secondName: "compiled-elementwise",
      first: eagerSeparateGateUp, second: compiledSeparateGateUp),
    benchmarkPair(
      workload: "mistral7b-gate-up", firstName: "two-op-compiled",
      secondName: "packed-row-fused",
      first: compiledSeparateGateUp, second: fusedGateUp),
    benchmarkPair(
      workload: "mistral7b-qkv", firstName: "three-op-current",
      secondName: "packed-row-fused", first: separateQKV, second: fusedQKV),
  ]

  print("Mistral-7B-shaped packed-row Metal A/B")
  print(
    "workload\tfirst\tsecond\tfirst_total_ms\tsecond_total_ms\t"
      + "second_total_speedup\tfirst_execution_ms\tsecond_execution_ms\t"
      + "second_execution_speedup"
  )
  for measurement in measurements {
    let firstTotal = median(measurement.firstTotalMilliseconds)
    let secondTotal = median(measurement.secondTotalMilliseconds)
    let firstExecution = median(measurement.firstExecutionMilliseconds)
    let secondExecution = median(measurement.secondExecutionMilliseconds)
    print(
      "\(measurement.workload)\t\(measurement.firstName)\t\(measurement.secondName)\t"
        + "\(format(firstTotal))\t\(format(secondTotal))\t"
        + "\(format(firstTotal / secondTotal))x\t"
        + "\(format(firstExecution))\t\(format(secondExecution))\t"
        + "\(format(firstExecution / secondExecution))x"
    )
  }
}

private func benchmarkPair(
  workload: String,
  firstName: String,
  secondName: String,
  first: (Int) -> [MLXArray],
  second: (Int) -> [MLXArray]
) -> PairedMeasurement {
  for index in 0..<warmupCount {
    eval(first(index))
    eval(second(index))
  }
  Stream.gpu.synchronize()

  var firstTotal: [Double] = []
  var secondTotal: [Double] = []
  var firstExecution: [Double] = []
  var secondExecution: [Double] = []
  let rounds = max(queueRounds, 5)

  func measure(
    _ operation: (Int) -> [MLXArray],
    round: Int
  ) -> (total: Double, execution: Double) {
    let totalStart = ContinuousClock.now
    let outputs = (0..<queueDepth).flatMap { index in
      operation(round * queueDepth + index)
    }
    let executionStart = ContinuousClock.now
    eval(outputs)
    Stream.gpu.synchronize()
    return (
      milliseconds(totalStart.duration(to: .now)) / Double(queueDepth),
      milliseconds(executionStart.duration(to: .now)) / Double(queueDepth)
    )
  }

  for round in 0..<rounds {
    if round.isMultiple(of: 2) {
      let a = measure(first, round: round)
      let b = measure(second, round: round)
      firstTotal.append(a.total)
      firstExecution.append(a.execution)
      secondTotal.append(b.total)
      secondExecution.append(b.execution)
    } else {
      let b = measure(second, round: round)
      let a = measure(first, round: round)
      secondTotal.append(b.total)
      secondExecution.append(b.execution)
      firstTotal.append(a.total)
      firstExecution.append(a.execution)
    }
  }

  return PairedMeasurement(
    workload: workload,
    firstName: firstName,
    secondName: secondName,
    firstTotalMilliseconds: firstTotal,
    secondTotalMilliseconds: secondTotal,
    firstExecutionMilliseconds: firstExecution,
    secondExecutionMilliseconds: secondExecution)
}

private func benchmarkAffineScaleSearch() {
  MLXRandom.seed(7)
  Memory.cacheLimit = 256 * 1_024 * 1_024

  let zeroSource = MLXArray.zeros([2, 64], dtype: .float32)
  let zeroBaseline = MLX.quantized(zeroSource, groupSize: 64, bits: 4, mode: .affine)
  let zeroSearched = q4AffineScaleSearchQuantized(zeroSource)
  precondition(
    MLX.arrayEqual(zeroSearched.weight, zeroBaseline.wq).item(Bool.self)
      && MLX.arrayEqual(zeroSearched.scales, zeroBaseline.scales).item(Bool.self)
      && MLX.arrayEqual(zeroSearched.biases, zeroBaseline.biases!).item(Bool.self),
    "scale search must preserve zero-group baseline arrays"
  )
  let stacked = q4AffineScaleSearchQuantized(
    MLXRandom.normal([3, 5, 64], dtype: .float32))
  precondition(
    stacked.weight.shape == [3, 5, 8]
      && stacked.scales.shape == [3, 5, 1]
      && stacked.biases.shape == [3, 5, 1],
    "scale search must preserve stacked expert layout"
  )

  // Keep the reported matrix deterministic independently of the correctness probes.
  MLXRandom.seed(7)

  let rows = 6_144
  let columns = 2_048
  let source = MLXRandom.normal([rows, columns], dtype: .bfloat16, scale: 0.02)
  let inputs = makeInputs([1, columns])

  // Compile and execute both conversion paths before timing. Otherwise a fresh process
  // charges Metal startup to the standard baseline merely because it runs first.
  do {
    let warmBaseline = MLX.quantized(source, groupSize: 64, bits: 4, mode: .affine)
    let warmSearched = q4AffineScaleSearchQuantized(source)
    evaluate(
      warmBaseline.wq,
      warmBaseline.scales,
      warmBaseline.biases,
      warmSearched.weight,
      warmSearched.scales,
      warmSearched.biases
    )
  }
  Memory.clearCache()

  let baselineStart = ContinuousClock.now
  let baseline = MLX.quantized(source, groupSize: 64, bits: 4, mode: .affine)
  evaluate(baseline.wq, baseline.scales, baseline.biases)
  let baselineConversionMS = milliseconds(baselineStart.duration(to: .now))

  let searchStart = ContinuousClock.now
  let searched = q4AffineScaleSearchQuantized(source)
  evaluate(searched.weight, searched.scales, searched.biases)
  let searchConversionMS = milliseconds(searchStart.duration(to: .now))

  let original = source.asType(.float32)
  let baselineRestored = MLX.dequantized(
    baseline.wq,
    scales: baseline.scales,
    biases: baseline.biases,
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
  let baselineMSE = MLX.mean(MLX.square(original - baselineRestored))
  let searchedMSE = MLX.mean(MLX.square(original - searchedRestored))
  evaluate(baselineMSE, searchedMSE)
  let baselineMSEValue = Double(baselineMSE.item(Float.self))
  let searchedMSEValue = Double(searchedMSE.item(Float.self))
  precondition(
    searchedMSEValue <= baselineMSEValue + 1e-12,
    "scale search increased reconstruction MSE"
  )

  func output(
    _ packed: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    input: MLXArray
  ) -> MLXArray {
    MLX.quantizedMM(
      input,
      packed,
      scales: scales,
      biases: biases,
      transpose: true,
      groupSize: 64,
      bits: 4,
      mode: .affine
    )
  }

  // Both variants use the same kernel specialization. Warm them together, then use ABBA
  // ordering so shader compilation, frequency ramp, and measurement order are not mistaken
  // for a runtime-format improvement.
  for index in 0..<max(20, warmupCount) {
    MLX.eval(
      output(
        baseline.wq,
        scales: baseline.scales,
        biases: baseline.biases!,
        input: inputs[index % inputs.count]
      ),
      output(
        searched.weight,
        scales: searched.scales,
        biases: searched.biases,
        input: inputs[index % inputs.count]
      )
    )
  }
  Stream.gpu.synchronize()

  let baselineA = benchmark { index in
    output(
      baseline.wq,
      scales: baseline.scales,
      biases: baseline.biases!,
      input: inputs[index % inputs.count]
    )
  }
  let searchedA = benchmark { index in
    output(
      searched.weight,
      scales: searched.scales,
      biases: searched.biases,
      input: inputs[index % inputs.count]
    )
  }
  let searchedB = benchmark { index in
    output(
      searched.weight,
      scales: searched.scales,
      biases: searched.biases,
      input: inputs[index % inputs.count]
    )
  }
  let baselineB = benchmark { index in
    output(
      baseline.wq,
      scales: baseline.scales,
      biases: baseline.biases!,
      input: inputs[index % inputs.count]
    )
  }
  let baselineQueued = median(baselineA.queued + baselineB.queued)
  let searchedQueued = median(searchedA.queued + searchedB.queued)
  let baselineBytes = baseline.wq.nbytes + baseline.scales.nbytes + baseline.biases!.nbytes
  let searchedBytes = searched.weight.nbytes + searched.scales.nbytes + searched.biases.nbytes
  precondition(baselineBytes == searchedBytes, "scale search changed stored byte count")

  print("AffineScaleSearch-Q4R8 Metal microbenchmark")
  print(
    "shape=\(rows)x\(columns) dtype=bfloat16 group=64 bits=4 "
      + "centered_scale_factors=0.75...1.25 step=0.0625 "
      + "bias_refinement=least_squares_1 joint_affine_refinement=least_squares_2"
  )
  print("metric\tstandard_q4\tscale_search_q4\trelative")
  print(
    "reconstruction_mse\t\(baselineMSEValue)\t\(searchedMSEValue)\t"
      + "\(format(baselineMSEValue / searchedMSEValue))x lower-is-better gain"
  )
  print(
    "conversion_ms\t\(format(baselineConversionMS))\t\(format(searchConversionMS))\t"
      + "\(format(searchConversionMS / baselineConversionMS))x overhead"
  )
  print(
    "queued_qmv_ms\t\(format(baselineQueued))\t\(format(searchedQueued))\t"
      + "\(format(baselineQueued / searchedQueued))x throughput"
  )
  print("stored_bytes\t\(baselineBytes)\t\(searchedBytes)\t\(baselineBytes == searchedBytes)")
}

private func benchmarkDense(
  _ quantization: QuantizationCase,
  workload: String = "dense-qmv",
  inputDimensions: Int,
  outputDimensions: Int
) -> Measurement {
  let inputs = makeInputs([1, inputDimensions])
  let weight = MLXRandom.normal(
    [outputDimensions, inputDimensions], dtype: .bfloat16, scale: 0.02)
  let packed = quantized(
    weight,
    groupSize: quantization.groupSize,
    bits: quantization.bits,
    mode: quantization.mode
  )
  eval(inputs)
  evaluate(packed.wq, packed.scales, packed.biases)

  let times = benchmark { index in
    quantizedMM(
      inputs[index % inputs.count],
      packed.wq,
      scales: packed.scales,
      biases: packed.biases,
      transpose: true,
      groupSize: quantization.groupSize,
      bits: quantization.bits,
      mode: quantization.mode
    )
  }
  return Measurement(
    workload: workload,
    quantization: quantization.name,
    synchronizedMedianMilliseconds: median(times.synchronized),
    synchronizedMinimumMilliseconds: times.synchronized.min()!,
    queuedMilliseconds: median(times.queued)
  )
}

private func benchmarkExpertGather(
  _ quantization: QuantizationCase,
  workload: String,
  inputDimensions: Int,
  outputDimensions: Int,
  numberOfExperts: Int,
  selectedExperts: Int
) -> Measurement {
  let inputs = makeInputs([1, 1, 1, 1, inputDimensions])
  let weight = MLXRandom.normal(
    [numberOfExperts, outputDimensions, inputDimensions],
    dtype: .bfloat16,
    scale: 0.02
  )
  let indices = makeExpertIndices(
    numberOfExperts: numberOfExperts,
    selectedExperts: selectedExperts
  )
  let packed = quantized(
    weight,
    groupSize: quantization.groupSize,
    bits: quantization.bits,
    mode: quantization.mode
  )
  eval(inputs)
  eval(indices)
  evaluate(packed.wq, packed.scales, packed.biases)

  let times = benchmark { index in
    gatherQuantizedMM(
      inputs[index % inputs.count],
      packed.wq,
      scales: packed.scales,
      biases: packed.biases,
      rhsIndices: indices[index % indices.count],
      transpose: true,
      groupSize: quantization.groupSize,
      bits: quantization.bits,
      mode: quantization.mode
    )
  }
  return Measurement(
    workload: workload,
    quantization: quantization.name,
    synchronizedMedianMilliseconds: median(times.synchronized),
    synchronizedMinimumMilliseconds: times.synchronized.min()!,
    queuedMilliseconds: median(times.queued)
  )
}

private func benchmarkSharedGateUpFusion(
  _ quantization: QuantizationCase
) -> (
  separate: (synchronized: [Double], queued: [Double]),
  fused: (synchronized: [Double], queued: [Double])
) {
  let inputDimensions = 2_048
  let hiddenDimensions = 512
  let inputs = makeInputs([1, inputDimensions])
  let gateWeight = MLXRandom.normal(
    [hiddenDimensions, inputDimensions], dtype: .bfloat16, scale: 0.02)
  let upWeight = MLXRandom.normal(
    [hiddenDimensions, inputDimensions], dtype: .bfloat16, scale: 0.02)
  let gate = quantized(
    gateWeight,
    groupSize: quantization.groupSize,
    bits: quantization.bits,
    mode: quantization.mode
  )
  let up = quantized(
    upWeight,
    groupSize: quantization.groupSize,
    bits: quantization.bits,
    mode: quantization.mode
  )
  let fusedWeight = concatenated([gate.wq, up.wq], axis: 0)
  let fusedScales = concatenated([gate.scales, up.scales], axis: 0)
  let fusedBiases = concatenated([gate.biases!, up.biases!], axis: 0)

  eval(inputs)
  evaluate(
    gate.wq, gate.scales, gate.biases,
    up.wq, up.scales, up.biases,
    fusedWeight, fusedScales, fusedBiases
  )

  func separateOutput(_ input: MLXArray) -> MLXArray {
    let gateOutput = quantizedMM(
      input,
      gate.wq,
      scales: gate.scales,
      biases: gate.biases,
      transpose: true,
      groupSize: quantization.groupSize,
      bits: quantization.bits,
      mode: quantization.mode
    )
    let upOutput = quantizedMM(
      input,
      up.wq,
      scales: up.scales,
      biases: up.biases,
      transpose: true,
      groupSize: quantization.groupSize,
      bits: quantization.bits,
      mode: quantization.mode
    )
    return siluProduct(gateOutput, upOutput)
  }

  func fusedOutput(_ input: MLXArray) -> MLXArray {
    let gateUp = quantizedMM(
      input,
      fusedWeight,
      scales: fusedScales,
      biases: fusedBiases,
      transpose: true,
      groupSize: quantization.groupSize,
      bits: quantization.bits,
      mode: quantization.mode
    )
    let parts = split(gateUp, parts: 2, axis: -1)
    return siluProduct(parts[0], parts[1])
  }

  precondition(
    allClose(
      separateOutput(inputs[0]), fusedOutput(inputs[0]),
      rtol: 1e-5, atol: 1e-5
    ).item(Bool.self),
    "fused gate/up output differs from the two-projection reference"
  )

  let separate = benchmark { index in
    separateOutput(inputs[index % inputs.count])
  }
  let fused = benchmark { index in
    fusedOutput(inputs[index % inputs.count])
  }

  return (separate, fused)
}

private func benchmark(_ operation: (Int) -> MLXArray) -> (
  synchronized: [Double], queued: [Double]
) {
  for _ in 0..<warmupCount {
    eval(operation(0))
  }
  Stream.gpu.synchronize()

  var synchronizedTimes: [Double] = []
  synchronizedTimes.reserveCapacity(iterationCount)
  for index in 0..<iterationCount {
    let start = ContinuousClock.now
    eval(operation(index))
    Stream.gpu.synchronize()
    synchronizedTimes.append(milliseconds(start.duration(to: .now)))
  }

  var queuedTimes: [Double] = []
  queuedTimes.reserveCapacity(queueRounds)
  for round in 0..<queueRounds {
    let outputs = (0..<queueDepth).map { index in
      operation(round * queueDepth + index)
    }
    let start = ContinuousClock.now
    eval(outputs)
    Stream.gpu.synchronize()
    queuedTimes.append(milliseconds(start.duration(to: .now)) / Double(queueDepth))
  }
  return (synchronizedTimes, queuedTimes)
}

private func makeInputs(_ shape: [Int]) -> [MLXArray] {
  let count = max(queueDepth, iterationCount)
  return (0..<count).map { _ in
    MLXRandom.normal(shape, dtype: .bfloat16)
  }
}

private func makeExpertIndices(
  numberOfExperts: Int,
  selectedExperts: Int
) -> [MLXArray] {
  let count = max(queueDepth, iterationCount)
  return (0..<count).map { group in
    let values = (0..<selectedExperts).map {
      (group * selectedExperts + $0) % numberOfExperts
    }
    return MLXArray(values, [1, 1, selectedExperts])
  }
}

private func evaluate(_ arrays: MLXArray?...) {
  eval(arrays.compactMap { $0 })
  Stream.gpu.synchronize()
}

private func milliseconds(_ duration: Duration) -> Double {
  let components = duration.components
  return Double(components.seconds) * 1_000
    + Double(components.attoseconds) / 1_000_000_000_000_000
}

private func median(_ values: [Double]) -> Double {
  let sorted = values.sorted()
  let middle = sorted.count / 2
  if sorted.count.isMultiple(of: 2) {
    return (sorted[middle - 1] + sorted[middle]) / 2
  }
  return sorted[middle]
}

private func format(_ value: Double) -> String {
  String(format: "%.3f", value)
}

private func formatPrecise(_ value: Double) -> String {
  String(format: "%.6f", value)
}

private func validateCommandLine() {
  let executableName = CommandLine.arguments.first?.split(separator: "/").last.map(String.init)
    ?? "wick-metal-quant-bench"
  let usage = """
    Usage: \(executableName) [MODE] [OPTIONS]
    With no MODE, run the existing Laguna-shaped quantization benchmark.

    Modes (choose at most one):
      --gpt-oss-20b           Synthetic GPT-OSS E32/top4 projections and expert operations.
      --format-ab            Laguna affine Q4 versus NVFP4.
      --scale-search-only    Affine scale-search reconstruction and timing.
      --laguna-graph-ab      Laguna graph fusion comparisons.
      --mistral-graph-ab     Mistral graph fusion comparisons.
      --qmv-specialization   Laguna affine QMV tile validation and timing.
      --router-precision-ab  Laguna router Q4 versus Q8.
      --fused-gather-silu-ab Laguna fused gather/SILU experiment.

    Options:
      --warmup N             Warmups, at least 1 (default 8).
      --iterations N         Synchronized iterations, at least 3 (default 40).
      --queue-depth N        Operations per queue round, at least 2 (default 32).
      --queue-rounds N       Queue rounds, at least 3 (default 7; paired modes use at least 5).
      --fused-gather-silu-candidate-first  Only with --fused-gather-silu-ab.
      --fused-gather-silu-output PATH      Only with --fused-gather-silu-ab.
      -h, --help             Print this help without starting a GPU benchmark.

    GPT-OSS mode uses random weights and reports primitive timings, not model tokens/s.
    Its K=2880 expert projections cannot use G128; the O projection K=4096 can.
    """
  let arguments = Array(CommandLine.arguments.dropFirst())
  if arguments.contains("--help") || arguments.contains("-h") {
    print(usage)
    exit(0)
  }
  func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("Error: \(message)\nUse --help for usage.\n".utf8))
    exit(2)
  }
  let modes: Set<String> = ["--gpt-oss-20b", "--format-ab", "--scale-search-only",
    "--laguna-graph-ab", "--mistral-graph-ab", "--qmv-specialization",
    "--router-precision-ab", "--fused-gather-silu-ab"]
  let minimums = ["--warmup": 1, "--iterations": 3, "--queue-depth": 2, "--queue-rounds": 3]
  let auxiliary = ["--fused-gather-silu-candidate-first", "--fused-gather-silu-output"]
  var seen = Set<String>()
  var index = 0
  while index < arguments.count {
    let argument = arguments[index]
    guard seen.insert(argument).inserted else { fail("duplicate option \(argument)") }
    if let minimum = minimums[argument] {
      index += 1
      guard index < arguments.count, let value = Int(arguments[index]), value >= minimum else {
        fail("\(argument) requires an integer of at least \(minimum)")
      }
    } else if argument == "--fused-gather-silu-output" {
      index += 1
      guard index < arguments.count, !arguments[index].isEmpty,
        !arguments[index].hasPrefix("--") else { fail("\(argument) requires an output path") }
    } else if !modes.contains(argument) && argument != "--fused-gather-silu-candidate-first" {
      fail("unknown option \(argument)")
    }
    index += 1
  }
  guard seen.intersection(modes).count <= 1 else { fail("choose only one benchmark mode") }
  if auxiliary.contains(where: { seen.contains($0) }) && !seen.contains("--fused-gather-silu-ab") {
    fail("fused gather/SILU options require --fused-gather-silu-ab")
  }
}

private func integerArgument(_ name: String, defaultValue: Int) -> Int {
  guard let index = CommandLine.arguments.firstIndex(of: name) else {
    return defaultValue
  }
  let valueIndex = CommandLine.arguments.index(after: index)
  guard valueIndex < CommandLine.arguments.endIndex,
    let value = Int(CommandLine.arguments[valueIndex])
  else {
    fatalError("\(name) requires an integer value")
  }
  return value
}

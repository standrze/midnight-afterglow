import Foundation
import MLX
import Testing

@testable import MistralActivationScaleSearchCore

@Suite("Expert activation-weighted ScaleSearch", .serialized)
struct ExpertActivationWeightedScaleSearchTests {
  @Test("Covered experts improve stored-grid objectives while under-covered experts retain bytes")
  func coverageAndObjective() throws {
    let values = (0..<(3 * 2 * 64)).map { Float(sin(Double($0) * 0.173) * 2.5) }
    let source = MLXArray(values).reshaped(3, 2, 64).asType(.bfloat16)
    let (weight, scales, optionalBiases) = MLX.quantized(source, groupSize: 64, bits: 4)
    let biases = try #require(optionalBiases)
    let moments = MLXArray((0..<(3 * 64)).map { Float(($0 % 11) + 1) }).reshaped(3, 64)
    let result = try MistralActivationWeightedScaleSearch.rescoreExperts(
      sourceWeight: source, templateWeight: weight, templateScales: scales, templateBiases: biases,
      calibrationSecondMoments: moments, calibrationPositionCounts: [100, 100, 0],
      validationSecondMoments: moments, validationPositionCounts: [100, 1, 0], minimumExpertPositions: 32)
    MLX.eval(result.weight, result.scales, result.biases)
    #expect(result.retainedTemplateExperts == [1, 2])
    #expect(result.weight.shape == weight.shape)
    #expect(result.weight.nbytes == weight.nbytes)
    #expect(result.scales.dtype == scales.dtype)
    #expect(result.biases.dtype == biases.dtype)
    for expert in [1, 2] {
      #expect(result.weight[expert].asArray(UInt32.self) == weight[expert].asArray(UInt32.self))
      #expect(result.scales[expert].asArray(Float.self) == scales[expert].asArray(Float.self))
      #expect(result.biases[expert].asArray(Float.self) == biases[expert].asArray(Float.self))
    }
    let diagnostic = try #require(result.diagnostics[0])
    #expect(diagnostic.candidateCalibrationWeightedMSE <= diagnostic.templateCalibrationWeightedMSE)
    let candidateValidation = try #require(diagnostic.candidateValidationWeightedMSE)
    let templateValidation = try #require(diagnostic.templateValidationWeightedMSE)
    #expect(candidateValidation <= templateValidation)
  }

  @Test("Missing expert channel coverage never becomes a fabricated uniform objective")
  func zeroMomentFallback() throws {
    let source = MLXArray.ones([2, 2, 64], type: Float.self)
    let (weight, scales, optionalBiases) = MLX.quantized(source, groupSize: 64, bits: 4)
    let biases = try #require(optionalBiases)
    let result = try MistralActivationWeightedScaleSearch.rescoreExperts(
      sourceWeight: source, templateWeight: weight, templateScales: scales, templateBiases: biases,
      calibrationSecondMoments: MLXArray.zeros([2, 64], type: Float.self),
      calibrationPositionCounts: [100, 100])
    #expect(result.retainedTemplateExperts == [0, 1])
    #expect(result.diagnostics.isEmpty)
    #expect(result.weight.asArray(UInt32.self) == weight.asArray(UInt32.self))
  }

  @Test("A dense activation vector cannot be silently reused for a stack of experts")
  func rejectsDenseProxy() throws {
    let source = MLXArray.ones([2, 2, 64], type: Float.self)
    let (weight, scales, optionalBiases) = MLX.quantized(source, groupSize: 64, bits: 4)
    let biases = try #require(optionalBiases)
    #expect(throws: ActivationWeightedScaleSearchError.self) {
      try MistralActivationWeightedScaleSearch.rescoreExperts(
        sourceWeight: source, templateWeight: weight, templateScales: scales, templateBiases: biases,
        calibrationSecondMoments: MLXArray.ones([64], type: Float.self), calibrationPositionCounts: [100, 100])
    }
  }
}

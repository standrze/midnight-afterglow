import Foundation
import MLX
import Testing

@testable import MistralActivationScaleSearchCore

@Suite("Mistral activation-weighted ScaleSearch")
struct MistralActivationScaleSearchCoreTests {
  @Test("Rescoring preserves the affine-Q4 layout and metadata dtypes")
  func preservesTemplateLayout() throws {
    let values = (0..<256).map { index in
      let position = Double(index)
      let first = sin(position * 0.173) * 2.5
      let second = cos(position * 0.071)
      return Float(first + second)
    }
    let source = MLXArray(values).reshaped(2, 128).asType(.bfloat16)
    let template = try standardAffineQ4(source)
    let moments = MLXArray((0..<128).map { Float(($0 % 11) + 1) })

    let result = try MistralActivationWeightedScaleSearch.rescore(
      sourceWeight: source,
      templateWeight: template.weight,
      templateScales: template.scales,
      templateBiases: template.biases,
      calibrationSecondMoments: moments
    )
    MLX.eval(result.weight, result.scales, result.biases)

    #expect(result.weight.shape == template.weight.shape)
    #expect(result.weight.dtype == DType.uint32)
    #expect(result.weight.nbytes == template.weight.nbytes)
    #expect(result.scales.shape == template.scales.shape)
    #expect(result.scales.dtype == template.scales.dtype)
    #expect(result.scales.nbytes == template.scales.nbytes)
    #expect(result.biases.shape == template.biases.shape)
    #expect(result.biases.dtype == template.biases.dtype)
    #expect(result.biases.nbytes == template.biases.nbytes)
    #expect(result.diagnostics.elementCount == source.size)
    #expect(result.diagnostics.groupCount == 4)
  }

  @Test("The LS2 template remains the exact fallback without a strict weighted gain")
  func retainsStrictFallback() throws {
    let source = MLXArray.zeros([2, 64], type: Float.self)
    let template = try standardAffineQ4(source)
    let moments = MLXArray((1...64).map(Float.init))

    let result = try MistralActivationWeightedScaleSearch.rescore(
      sourceWeight: source,
      templateWeight: template.weight,
      templateScales: template.scales,
      templateBiases: template.biases,
      calibrationSecondMoments: moments
    )
    MLX.eval(result.weight, result.scales, result.biases)

    #expect(result.diagnostics.changedGroupCount == 0)
    #expect(result.diagnostics.templateCalibrationWeightedMSE == 0)
    #expect(result.diagnostics.candidateCalibrationWeightedMSE == 0)
    #expect(result.weight.asArray(UInt32.self) == template.weight.asArray(UInt32.self))
    #expect(result.scales.asArray(Float.self) == template.scales.asArray(Float.self))
    #expect(result.biases.asArray(Float.self) == template.biases.asArray(Float.self))
  }

  @Test("Activation importance changes a deliberately poor Q4 group")
  func changesCraftedGroup() throws {
    let fixture = craftedOutlierFixture()

    let result = try MistralActivationWeightedScaleSearch.rescore(
      sourceWeight: fixture.source,
      templateWeight: fixture.weight,
      templateScales: fixture.scales,
      templateBiases: fixture.biases,
      calibrationSecondMoments: fixture.calibration
    )

    #expect(result.diagnostics.changedGroupCount == 1)
    #expect(
      result.diagnostics.candidateCalibrationWeightedMSE
        < result.diagnostics.templateCalibrationWeightedMSE
    )
  }

  @Test("Invalid channel statistics fail closed")
  func rejectsInvalidMoments() {
    let fixture = craftedOutlierFixture()
    var negativeValues = [Float](repeating: 1, count: 64)
    negativeValues[17] = -1

    #expect(throws: ActivationWeightedScaleSearchError.self) {
      try MistralActivationWeightedScaleSearch.rescore(
        sourceWeight: fixture.source,
        templateWeight: fixture.weight,
        templateScales: fixture.scales,
        templateBiases: fixture.biases,
        calibrationSecondMoments: MLXArray(negativeValues)
      )
    }
    #expect(throws: ActivationWeightedScaleSearchError.self) {
      try MistralActivationWeightedScaleSearch.rescore(
        sourceWeight: fixture.source,
        templateWeight: fixture.weight,
        templateScales: fixture.scales,
        templateBiases: fixture.biases,
        calibrationSecondMoments: MLXArray.zeros([64], type: Float.self)
      )
    }
  }

  @Test("Held-out moments can veto a calibration-only improvement")
  func validationGuardRejectsRegression() throws {
    let fixture = craftedOutlierFixture()
    let calibrationOnly = try MistralActivationWeightedScaleSearch.rescore(
      sourceWeight: fixture.source,
      templateWeight: fixture.weight,
      templateScales: fixture.scales,
      templateBiases: fixture.biases,
      calibrationSecondMoments: fixture.calibration
    )
    let guarded = try MistralActivationWeightedScaleSearch.rescore(
      sourceWeight: fixture.source,
      templateWeight: fixture.weight,
      templateScales: fixture.scales,
      templateBiases: fixture.biases,
      calibrationSecondMoments: fixture.calibration,
      validationSecondMoments: fixture.validation
    )

    #expect(calibrationOnly.diagnostics.changedGroupCount == 1)
    #expect(guarded.diagnostics.changedGroupCount == 0)
    #expect(guarded.diagnostics.validationRejectedGroupCount == 1)
    #expect(guarded.diagnostics.templateValidationWeightedMSE == 0)
    #expect(guarded.diagnostics.candidateValidationWeightedMSE == 0)
    #expect(guarded.weight.asArray(UInt32.self) == fixture.weight.asArray(UInt32.self))
  }

  private func standardAffineQ4(
    _ source: MLXArray
  ) throws -> (weight: MLXArray, scales: MLXArray, biases: MLXArray) {
    let result = MLX.quantized(source, groupSize: 64, bits: 4, mode: .affine)
    guard let biases = result.biases else {
      throw ActivationWeightedScaleSearchError.invalidTemplate(
        "MLX affine Q4 unexpectedly omitted biases")
    }
    MLX.eval(result.wq, result.scales, biases)
    return (result.wq, result.scales, biases)
  }

  private func craftedOutlierFixture() -> (
    source: MLXArray,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    calibration: MLXArray,
    validation: MLXArray
  ) {
    var sourceValues = [Float](repeating: 0, count: 64)
    sourceValues[63] = 10.3
    var calibrationValues = [Float](repeating: 0, count: 64)
    calibrationValues[63] = 1
    var validationValues = [Float](repeating: 1, count: 64)
    validationValues[63] = 0
    return (
      source: MLXArray(sourceValues).reshaped(1, 64),
      weight: MLXArray.zeros([1, 8], type: UInt32.self),
      scales: MLXArray.ones([1, 1], type: Float.self),
      biases: MLXArray.zeros([1, 1], type: Float.self),
      calibration: MLXArray(calibrationValues),
      validation: MLXArray(validationValues)
    )
  }
}

import MLX

/// A stacked affine-Q4 result whose under-covered experts retain every template
/// weight code, scale and bias exactly. Diagnostics exist only for searched experts.
public struct ExpertActivationWeightedScaleSearchResult {
  public var weight: MLXArray
  public var scales: MLXArray
  public var biases: MLXArray
  public var diagnostics: [Int: ActivationWeightedScaleSearchDiagnostics]
  public var retainedTemplateExperts: [Int]
}

extension MistralActivationWeightedScaleSearch {
  /// Refine each [output, input] expert using its own conditional input moments.
  /// This deliberately does not broadcast a dense activation vector over experts.
  /// The fused gate/up output dimension is preserved as one calibration unit.
  public static func rescoreExperts(
    sourceWeight: MLXArray,
    templateWeight: MLXArray,
    templateScales: MLXArray,
    templateBiases: MLXArray,
    calibrationSecondMoments: MLXArray,
    calibrationPositionCounts: [Int],
    validationSecondMoments: MLXArray? = nil,
    validationPositionCounts: [Int]? = nil,
    minimumExpertPositions: Int = 32
  ) throws -> ExpertActivationWeightedScaleSearchResult {
    guard sourceWeight.ndim == 3, sourceWeight.shape.allSatisfy({ $0 > 0 }),
      sourceWeight.dtype.isFloatingPoint, !sourceWeight.dtype.isComplex
    else {
      throw ActivationWeightedScaleSearchError.invalidSource("expected [experts, outputs, inputs] floating-point weights")
    }
    let experts = sourceWeight.dim(0)
    let outputs = sourceWeight.dim(1)
    let width = sourceWeight.dim(2)
    guard width % groupSize == 0, minimumExpertPositions > 0,
      templateWeight.shape == [experts, outputs, width / 8], templateWeight.dtype == .uint32,
      templateScales.shape == [experts, outputs, width / groupSize],
      templateBiases.shape == templateScales.shape,
      templateScales.dtype == templateBiases.dtype, templateScales.dtype.isFloatingPoint,
      calibrationSecondMoments.shape == [experts, width],
      calibrationPositionCounts.count == experts,
      calibrationPositionCounts.allSatisfy({ $0 >= 0 }),
      (validationSecondMoments == nil) == (validationPositionCounts == nil)
    else {
      throw ActivationWeightedScaleSearchError.invalidTemplate("incompatible expert quantization geometry or coverage")
    }
    if let validationSecondMoments, let validationPositionCounts {
      guard validationSecondMoments.shape == [experts, width],
        validationPositionCounts.count == experts,
        validationPositionCounts.allSatisfy({ $0 >= 0 })
      else {
        throw ActivationWeightedScaleSearchError.invalidMoments("incompatible validation expert geometry or counts")
      }
    }
    var weights = [MLXArray]()
    var scales = [MLXArray]()
    var biases = [MLXArray]()
    var diagnostics = [Int: ActivationWeightedScaleSearchDiagnostics]()
    var retained = [Int]()
    for expert in 0..<experts {
      let covered = calibrationPositionCounts[expert] >= minimumExpertPositions
        && (validationPositionCounts.map { $0[expert] >= minimumExpertPositions } ?? true)
      guard covered else {
        weights.append(templateWeight[expert])
        scales.append(templateScales[expert])
        biases.append(templateBiases[expert])
        retained.append(expert)
        continue
      }
      let calibration = calibrationSecondMoments[expert]
      let validation = validationSecondMoments.map { $0[expert] }
      // A silent channel group supplies no fitting objective. Retain this
      // expert instead of inventing uniform importance for unobserved channels.
      var hasPositiveGroups = true
      for moments in [calibration, validation].compactMap({ $0 }) {
        try MLX.checkedEval(moments)
        let values = moments.asType(.float32).asArray(Float.self)
        guard values.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
          throw ActivationWeightedScaleSearchError.invalidMoments("non-finite or negative moments for expert \(expert)")
        }
        for start in stride(from: 0, to: width, by: groupSize) {
          if !values[start..<(start + groupSize)].contains(where: { $0 > 0 }) {
            hasPositiveGroups = false
          }
        }
      }
      guard hasPositiveGroups else {
        weights.append(templateWeight[expert])
        scales.append(templateScales[expert])
        biases.append(templateBiases[expert])
        retained.append(expert)
        continue
      }
      let result = try rescore(
        sourceWeight: sourceWeight[expert],
        templateWeight: templateWeight[expert],
        templateScales: templateScales[expert],
        templateBiases: templateBiases[expert],
        calibrationSecondMoments: calibration,
        validationSecondMoments: validation)
      weights.append(result.weight)
      scales.append(result.scales)
      biases.append(result.biases)
      diagnostics[expert] = result.diagnostics
    }
    return ExpertActivationWeightedScaleSearchResult(
      weight: MLX.stacked(weights), scales: MLX.stacked(scales), biases: MLX.stacked(biases),
      diagnostics: diagnostics, retainedTemplateExperts: retained)
  }
}

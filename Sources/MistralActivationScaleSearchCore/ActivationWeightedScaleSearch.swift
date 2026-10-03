import Foundation
import MLX

/// Summary of one dense matrix's activation-weighted affine-Q4 second pass.
///
/// The weighted errors use input-channel second moments normalized to mean one
/// inside each 64-value quantization group. They are therefore comparable to
/// ordinary MSE while retaining the diagonal expected-output-error objective.
public struct ActivationWeightedScaleSearchDiagnostics: Codable, Equatable, Sendable {
  public var elementCount: Int
  public var groupCount: Int
  public var changedGroupCount: Int
  public var validationRejectedGroupCount: Int
  public var templateCalibrationWeightedMSE: Double
  public var candidateCalibrationWeightedMSE: Double
  public var calibrationWeightedMSEReductionPercent: Double
  public var templateValidationWeightedMSE: Double?
  public var candidateValidationWeightedMSE: Double?
  public var validationWeightedMSEReductionPercent: Double?
  public var templateRawMSE: Double
  public var candidateRawMSE: Double
  public var rawMSEReductionPercent: Double

  enum CodingKeys: String, CodingKey {
    case elementCount = "element_count"
    case groupCount = "group_count"
    case changedGroupCount = "changed_group_count"
    case validationRejectedGroupCount = "validation_rejected_group_count"
    case templateCalibrationWeightedMSE = "template_calibration_weighted_mse"
    case candidateCalibrationWeightedMSE = "candidate_calibration_weighted_mse"
    case calibrationWeightedMSEReductionPercent =
      "calibration_weighted_mse_reduction_percent"
    case templateValidationWeightedMSE = "template_validation_weighted_mse"
    case candidateValidationWeightedMSE = "candidate_validation_weighted_mse"
    case validationWeightedMSEReductionPercent =
      "validation_weighted_mse_reduction_percent"
    case templateRawMSE = "template_raw_mse"
    case candidateRawMSE = "candidate_raw_mse"
    case rawMSEReductionPercent = "raw_mse_reduction_percent"
  }
}

public struct ActivationWeightedScaleSearchResult {
  public var weight: MLXArray
  public var scales: MLXArray
  public var biases: MLXArray
  public var diagnostics: ActivationWeightedScaleSearchDiagnostics
}

public enum ActivationWeightedScaleSearchError: Error, LocalizedError, Equatable, Sendable {
  case invalidSource(String)
  case invalidTemplate(String)
  case invalidMoments(String)

  public var errorDescription: String? {
    switch self {
    case .invalidSource(let message):
      "Invalid BF16 source matrix: \(message)"
    case .invalidTemplate(let message):
      "Invalid LS2 template matrix: \(message)"
    case .invalidMoments(let message):
      "Invalid activation second moments: \(message)"
    }
  }
}

/// A Mistral-only, conversion-time refinement of an existing LS2 affine-Q4 grid.
///
/// For a linear projection `y = x W^T`, the diagonal approximation to expected
/// output error is `sum_j E[x_j^2] * (W_ij - Q_ij)^2`. This implementation uses
/// that objective for candidate selection, the fixed-slope bias fit, and both
/// joint slope/intercept fits. The supplied LS2 arrays initialize every group,
/// so the returned matrix cannot be worse than the template on the calibration
/// objective. When validation moments are supplied, a group is changed only if
/// it also does not worsen the held-out diagonal objective.
///
/// The result retains the template's packed UInt32 codes, scale/bias dtypes,
/// shapes, affine Q4 group-64 geometry, and ordinary runtime kernels.
public enum MistralActivationWeightedScaleSearch {
  public static let bits = 4
  public static let groupSize = 64
  public static let searchFactors: [Float] = [
    0.75, 0.8125, 0.875, 0.9375, 1,
    1.0625, 1.125, 1.1875, 1.25,
  ]

  public static func rescore(
    sourceWeight: MLXArray,
    templateWeight: MLXArray,
    templateScales: MLXArray,
    templateBiases: MLXArray,
    calibrationSecondMoments: MLXArray,
    validationSecondMoments: MLXArray? = nil
  ) throws -> ActivationWeightedScaleSearchResult {
    let inputWidth = try validateGeometry(
      sourceWeight: sourceWeight,
      templateWeight: templateWeight,
      templateScales: templateScales,
      templateBiases: templateBiases
    )
    let calibrationValues = try normalizedMoments(
      calibrationSecondMoments,
      inputWidth: inputWidth,
      label: "calibration"
    )
    let validationValues = try validationSecondMoments.map {
      try normalizedMoments($0, inputWidth: inputWidth, label: "validation")
    }

    let groupsPerRow = inputWidth / groupSize
    let leadingCount = sourceWeight.size / inputWidth
    let groupCount = leadingCount * groupsPerRow
    let wordsPerGroup = groupSize * bits / 32
    let codesPerWord = 32 / bits

    let grouped = sourceWeight.reshaped(groupCount, groupSize).asType(.float32)
    let calibrationWeights = groupedMoments(
      calibrationValues,
      leadingCount: leadingCount,
      groupsPerRow: groupsPerRow
    )
    let validationWeights = validationValues.map {
      groupedMoments($0, leadingCount: leadingCount, groupsPerRow: groupsPerRow)
    }

    let baselineWeight = templateWeight.reshaped(groupCount, wordsPerGroup)
    let baselineScales = templateScales.reshaped(groupCount, 1).asType(.float32)
    let baselineBiases = templateBiases.reshaped(groupCount, 1).asType(.float32)
    let shifts =
      (MLXArray(0..<codesPerWord) * bits)
      .asType(.uint32)
      .reshaped(1, 1, codesPerWord)
    let baselineWords = baselineWeight.reshaped(groupCount, wordsPerGroup, 1)
    let baselineCodes = MLX.bitwiseAnd(
      baselineWords >> shifts,
      MLXArray(UInt32((1 << bits) - 1))
    ).reshaped(groupCount, groupSize)
    let baselineReconstruction =
      baselineCodes.asType(.float32) * baselineScales + baselineBiases

    func weightedMean(_ value: MLXArray, _ weights: MLXArray) -> MLXArray {
      let numerator = MLX.sum(value * weights, axis: -1, keepDims: true)
      let denominator = MLX.sum(weights, axis: -1, keepDims: true)
      return numerator / denominator
    }

    var bestCalibrationError = weightedMean(
      MLX.square(grouped - baselineReconstruction), calibrationWeights)
    var bestValidationError = validationWeights.map {
      weightedMean(MLX.square(grouped - baselineReconstruction), $0)
    }
    var bestRawError = MLX.mean(
      MLX.square(grouped - baselineReconstruction), axis: -1, keepDims: true)
    let templateCalibrationError = bestCalibrationError
    let templateValidationError = bestValidationError
    let templateRawError = bestRawError

    var bestWeight = baselineWeight
    var bestScales = baselineScales
    var bestBiases = baselineBiases
    var validationRejected = MLXArray.zeros([groupCount, 1], type: Bool.self)
    var initialValues = [
      bestCalibrationError, bestRawError, bestWeight, bestScales, bestBiases,
    ]
    if let bestValidationError { initialValues.append(bestValidationError) }
    MLX.eval(initialValues)

    let baselineCenter = baselineBiases + baselineScales * Float(7.5)
    let calibrationGroupMean = weightedMean(grouped, calibrationWeights)

    for factor in searchFactors {
      // Round-trip scale and bias through the exact stored metadata dtype before
      // assigning codes or measuring either objective.
      let candidateScale =
        (baselineScales * factor).asType(templateScales.dtype).asType(.float32)
      let centeredBias =
        (baselineCenter - candidateScale * Float(7.5))
        .asType(templateBiases.dtype)
        .asType(.float32)
      let valid = MLX.abs(candidateScale) .> Float(0)
      let safeScale = MLX.where(valid, candidateScale, baselineScales)
      let safeCenteredBias = MLX.where(valid, centeredBias, baselineBiases)
      let centeredCodes = MLX.clip(
        MLX.round((grouped - safeCenteredBias) / safeScale), min: 0, max: 15
      ).asType(.uint32)
      let centeredReconstruction =
        centeredCodes.asType(.float32) * safeScale + safeCenteredBias
      let centeredCalibrationError = weightedMean(
        MLX.square(grouped - centeredReconstruction), calibrationWeights)

      // With scale and integer assignments fixed, the activation-weighted mean
      // residual is the exact intercept for the diagonal objective.
      let refinedBias = weightedMean(
        grouped - centeredCodes.asType(.float32) * safeScale,
        calibrationWeights
      ).asType(templateBiases.dtype).asType(.float32)
      let safeRefinedBias = MLX.where(valid, refinedBias, baselineBiases)
      let refinedCodes = MLX.clip(
        MLX.round((grouped - safeRefinedBias) / safeScale), min: 0, max: 15
      ).asType(.uint32)
      let refinedReconstruction =
        refinedCodes.asType(.float32) * safeScale + safeRefinedBias
      let refinedCalibrationError = weightedMean(
        MLX.square(grouped - refinedReconstruction), calibrationWeights)
      let useRefined = valid .&& (refinedCalibrationError .< centeredCalibrationError)
      let biasRefinedBias = MLX.where(useRefined, safeRefinedBias, safeCenteredBias)
      let biasRefinedCodes = MLX.where(useRefined, refinedCodes, centeredCodes)
      let biasRefinedCalibrationError = MLX.where(
        useRefined, refinedCalibrationError, centeredCalibrationError)

      // First activation-weighted joint slope/intercept fit.
      let codeValues = biasRefinedCodes.asType(.float32)
      let codeMean = weightedMean(codeValues, calibrationWeights)
      let centeredCodesForFit = codeValues - codeMean
      let codeVariance = weightedMean(
        MLX.square(centeredCodesForFit), calibrationWeights)
      let hasVariance = codeVariance .> Float(0)
      let safeVariance = MLX.where(hasVariance, codeVariance, MLXArray(Float(1)))
      let covariance = weightedMean(
        centeredCodesForFit * (grouped - calibrationGroupMean),
        calibrationWeights
      )
      let fittedScale =
        (covariance / safeVariance).asType(templateScales.dtype).asType(.float32)
      let fittedBias =
        (calibrationGroupMean - fittedScale * codeMean)
        .asType(templateBiases.dtype)
        .asType(.float32)
      let fitValid =
        valid .&& hasVariance
        .&& (MLX.abs(fittedScale) .> Float(0))
        .&& ((fittedScale * baselineScales) .> Float(0))
      let safeFittedScale = MLX.where(fitValid, fittedScale, safeScale)
      let safeFittedBias = MLX.where(fitValid, fittedBias, biasRefinedBias)
      let fittedCodes = MLX.clip(
        MLX.round((grouped - safeFittedBias) / safeFittedScale), min: 0, max: 15
      ).asType(.uint32)
      let fittedReconstruction =
        fittedCodes.asType(.float32) * safeFittedScale + safeFittedBias
      let fittedCalibrationError = weightedMean(
        MLX.square(grouped - fittedReconstruction), calibrationWeights)
      let useFitted = fitValid .&& (fittedCalibrationError .< biasRefinedCalibrationError)
      let selectedScale = MLX.where(useFitted, safeFittedScale, safeScale)
      let candidateBias = MLX.where(useFitted, safeFittedBias, biasRefinedBias)
      let candidateCodes = MLX.where(useFitted, fittedCodes, biasRefinedCodes)
      let candidateCalibrationError = MLX.where(
        useFitted, fittedCalibrationError, biasRefinedCalibrationError)

      // A second coordinate-descent step captures groups whose assignments
      // changed after the first weighted joint fit.
      let secondCodeValues = candidateCodes.asType(.float32)
      let secondCodeMean = weightedMean(secondCodeValues, calibrationWeights)
      let secondCenteredCodes = secondCodeValues - secondCodeMean
      let secondCodeVariance = weightedMean(
        MLX.square(secondCenteredCodes), calibrationWeights)
      let secondHasVariance = secondCodeVariance .> Float(0)
      let secondSafeVariance = MLX.where(
        secondHasVariance, secondCodeVariance, MLXArray(Float(1)))
      let secondCovariance = weightedMean(
        secondCenteredCodes * (grouped - calibrationGroupMean),
        calibrationWeights
      )
      let secondFittedScale =
        (secondCovariance / secondSafeVariance)
        .asType(templateScales.dtype)
        .asType(.float32)
      let secondFittedBias =
        (calibrationGroupMean - secondFittedScale * secondCodeMean)
        .asType(templateBiases.dtype)
        .asType(.float32)
      let secondFitValid =
        valid .&& secondHasVariance
        .&& (MLX.abs(secondFittedScale) .> Float(0))
        .&& ((secondFittedScale * baselineScales) .> Float(0))
      let safeSecondScale = MLX.where(secondFitValid, secondFittedScale, selectedScale)
      let safeSecondBias = MLX.where(secondFitValid, secondFittedBias, candidateBias)
      let secondFittedCodes = MLX.clip(
        MLX.round((grouped - safeSecondBias) / safeSecondScale), min: 0, max: 15
      ).asType(.uint32)
      let secondReconstruction =
        secondFittedCodes.asType(.float32) * safeSecondScale + safeSecondBias
      let secondCalibrationError = weightedMean(
        MLX.square(grouped - secondReconstruction), calibrationWeights)
      let useSecondFit =
        secondFitValid
        .&& (secondCalibrationError .< candidateCalibrationError)
      let finalScale = MLX.where(useSecondFit, safeSecondScale, selectedScale)
      let finalBias = MLX.where(useSecondFit, safeSecondBias, candidateBias)
      let finalCodes = MLX.where(useSecondFit, secondFittedCodes, candidateCodes)
      let finalCalibrationError = MLX.where(
        useSecondFit, secondCalibrationError, candidateCalibrationError)
      let finalReconstruction =
        finalCodes.asType(.float32) * finalScale + finalBias
      let finalRawError = MLX.mean(
        MLX.square(grouped - finalReconstruction), axis: -1, keepDims: true)

      let calibrationImproved = valid .&& (finalCalibrationError .< bestCalibrationError)
      let improved: MLXArray
      let finalValidationError: MLXArray?
      if let validationWeights, let currentValidationError = bestValidationError {
        let validationError = weightedMean(
          MLX.square(grouped - finalReconstruction), validationWeights)
        let validationNonWorse = validationError .<= currentValidationError
        validationRejected = logicalOr(
          validationRejected,
          calibrationImproved .&& logicalNot(validationNonWorse)
        )
        improved = (calibrationImproved .&& validationNonWorse)
        finalValidationError = validationError
      } else {
        improved = calibrationImproved
        finalValidationError = nil
      }

      let packedGroups = finalCodes.reshaped(groupCount, wordsPerGroup, codesPerWord)
      let candidateWeight = MLX.sum(packedGroups << shifts, axis: -1).asType(.uint32)
      bestWeight = MLX.where(improved, candidateWeight, bestWeight)
      bestScales = MLX.where(improved, finalScale, bestScales)
      bestBiases = MLX.where(improved, finalBias, bestBiases)
      bestCalibrationError = MLX.where(
        improved, finalCalibrationError, bestCalibrationError)
      bestRawError = MLX.where(improved, finalRawError, bestRawError)
      if let finalValidationError, let currentValidationError = bestValidationError {
        bestValidationError = MLX.where(
          improved, finalValidationError, currentValidationError)
      }

      // This is an offline conversion. Bound the lazy graph after every search
      // factor so the 131k-vocabulary lm_head remains practical on a 64 GB Mac.
      var liveValues = [
        bestWeight, bestScales, bestBiases, bestCalibrationError, bestRawError,
        validationRejected,
      ]
      if let bestValidationError { liveValues.append(bestValidationError) }
      MLX.eval(liveValues)
    }

    let weightChanged = MLX.any(
      bestWeight .!= baselineWeight, axis: -1, keepDims: true)
    let metadataChanged = logicalOr(
      bestScales .!= baselineScales,
      bestBiases .!= baselineBiases)
    let changedGroups = MLX.sum(
      logicalOr(weightChanged, metadataChanged).asType(.int32))
    let validationRejectedGroups = MLX.sum(validationRejected.asType(.int32))

    let templateCalibrationMean = MLX.mean(templateCalibrationError)
    let candidateCalibrationMean = MLX.mean(bestCalibrationError)
    let templateRawMean = MLX.mean(templateRawError)
    let candidateRawMean = MLX.mean(bestRawError)
    let templateValidationMean = templateValidationError.map { MLX.mean($0) }
    let candidateValidationMean = bestValidationError.map { MLX.mean($0) }
    var diagnosticArrays = [
      templateCalibrationMean, candidateCalibrationMean,
      templateRawMean, candidateRawMean,
      changedGroups, validationRejectedGroups,
    ]
    if let templateValidationMean { diagnosticArrays.append(templateValidationMean) }
    if let candidateValidationMean { diagnosticArrays.append(candidateValidationMean) }
    MLX.eval(diagnosticArrays)
    Stream.defaultStream(Device.defaultDevice()).synchronize()

    let templateCalibration = Double(templateCalibrationMean.item(Float.self))
    let candidateCalibration = Double(candidateCalibrationMean.item(Float.self))
    let templateRaw = Double(templateRawMean.item(Float.self))
    let candidateRaw = Double(candidateRawMean.item(Float.self))
    let templateValidation = templateValidationMean.map {
      Double($0.item(Float.self))
    }
    let candidateValidation = candidateValidationMean.map {
      Double($0.item(Float.self))
    }
    let validationReduction: Double?
    if let templateValidation, let candidateValidation {
      validationReduction = percentReduction(
        from: templateValidation, to: candidateValidation)
    } else {
      validationReduction = nil
    }
    let diagnostics = ActivationWeightedScaleSearchDiagnostics(
      elementCount: sourceWeight.size,
      groupCount: groupCount,
      changedGroupCount: Int(changedGroups.item(Int32.self)),
      validationRejectedGroupCount: Int(validationRejectedGroups.item(Int32.self)),
      templateCalibrationWeightedMSE: templateCalibration,
      candidateCalibrationWeightedMSE: candidateCalibration,
      calibrationWeightedMSEReductionPercent: percentReduction(
        from: templateCalibration, to: candidateCalibration),
      templateValidationWeightedMSE: templateValidation,
      candidateValidationWeightedMSE: candidateValidation,
      validationWeightedMSEReductionPercent: validationReduction,
      templateRawMSE: templateRaw,
      candidateRawMSE: candidateRaw,
      rawMSEReductionPercent: percentReduction(from: templateRaw, to: candidateRaw)
    )

    return ActivationWeightedScaleSearchResult(
      weight: bestWeight.reshaped(templateWeight.shape),
      scales: bestScales.reshaped(templateScales.shape).asType(templateScales.dtype),
      biases: bestBiases.reshaped(templateBiases.shape).asType(templateBiases.dtype),
      diagnostics: diagnostics
    )
  }

  private static func validateGeometry(
    sourceWeight: MLXArray,
    templateWeight: MLXArray,
    templateScales: MLXArray,
    templateBiases: MLXArray
  ) throws -> Int {
    guard sourceWeight.ndim == 2 else {
      throw ActivationWeightedScaleSearchError.invalidSource(
        "expected a dense rank-2 Linear weight, got shape \(sourceWeight.shape)")
    }
    guard sourceWeight.dtype.isFloatingPoint, !sourceWeight.dtype.isComplex else {
      throw ActivationWeightedScaleSearchError.invalidSource(
        "expected a real floating-point weight, got \(sourceWeight.dtype)")
    }
    let inputWidth = sourceWeight.dim(-1)
    guard inputWidth > 0, inputWidth % groupSize == 0 else {
      throw ActivationWeightedScaleSearchError.invalidSource(
        "input width \(inputWidth) is not divisible by \(groupSize)")
    }
    var expectedWeightShape = sourceWeight.shape
    expectedWeightShape[expectedWeightShape.count - 1] = inputWidth * bits / 32
    var expectedMetadataShape = sourceWeight.shape
    expectedMetadataShape[expectedMetadataShape.count - 1] = inputWidth / groupSize
    guard templateWeight.shape == expectedWeightShape, templateWeight.dtype == .uint32 else {
      throw ActivationWeightedScaleSearchError.invalidTemplate(
        "packed weight expected shape \(expectedWeightShape) UInt32, got "
          + "\(templateWeight.shape) \(templateWeight.dtype)")
    }
    guard templateScales.shape == expectedMetadataShape,
      templateBiases.shape == expectedMetadataShape,
      templateScales.dtype == sourceWeight.dtype,
      templateBiases.dtype == sourceWeight.dtype
    else {
      throw ActivationWeightedScaleSearchError.invalidTemplate(
        "scales/biases expected shape \(expectedMetadataShape) and dtype "
          + "\(sourceWeight.dtype), got scales \(templateScales.shape) "
          + "\(templateScales.dtype), biases \(templateBiases.shape) "
          + "\(templateBiases.dtype)")
    }
    return inputWidth
  }

  /// Validates raw E[x^2] and normalizes every quantization group to mean one.
  /// Multiplying a whole group's objective by a positive constant cannot alter
  /// its selected grid; normalization only improves numerical conditioning.
  private static func normalizedMoments(
    _ moments: MLXArray,
    inputWidth: Int,
    label: String
  ) throws -> [Float] {
    guard moments.ndim == 1, moments.shape == [inputWidth], moments.dtype == .float32 else {
      throw ActivationWeightedScaleSearchError.invalidMoments(
        "\(label) expected rank-1 Float32 [\(inputWidth)], got "
          + "\(moments.shape) \(moments.dtype)")
    }
    let values = moments.asArray(Float.self)
    var normalized = values
    for groupStart in stride(from: 0, to: inputWidth, by: groupSize) {
      var sum: Double = 0
      for offset in 0..<groupSize {
        let value = values[groupStart + offset]
        guard value.isFinite, value >= 0 else {
          throw ActivationWeightedScaleSearchError.invalidMoments(
            "\(label) channel \(groupStart + offset) must be finite and nonnegative")
        }
        sum += Double(value)
      }
      guard sum.isFinite, sum > 0 else {
        throw ActivationWeightedScaleSearchError.invalidMoments(
          "\(label) group \(groupStart / groupSize) has no positive mass")
      }
      let multiplier = Float(Double(groupSize) / sum)
      for offset in 0..<groupSize {
        normalized[groupStart + offset] *= multiplier
      }
    }
    return normalized
  }

  private static func groupedMoments(
    _ normalized: [Float],
    leadingCount: Int,
    groupsPerRow: Int
  ) -> MLXArray {
    let oneRow = MLXArray(normalized).reshaped(1, groupsPerRow, groupSize)
    return MLX.broadcast(
      oneRow, to: [leadingCount, groupsPerRow, groupSize]
    ).reshaped(leadingCount * groupsPerRow, groupSize)
  }

  private static func percentReduction(from baseline: Double, to candidate: Double) -> Double {
    guard baseline.isFinite, candidate.isFinite, baseline > 0 else { return 0 }
    return (baseline - candidate) / baseline * 100
  }
}

import Foundation
import MLX

public struct GemmaActivationWeightedScaleSearchResult {
    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray
    /// Dense projections use key zero; routed projections use expert indices.
    public let diagnostics: [Int: ActivationWeightedScaleSearchDiagnostics]
    public let retainedTemplateExperts: [Int]
    public let retainedModuleReason: String?
}

/// Projection fitting for native Gemma 4 affine Q4/G64 templates. The checkpoint
/// driver must supply tensors from the independently validated BF16 source and
/// actual template, and preserve every unselected tensor and all sidecars.
/// This fits a diagonal output-error proxy, not model-level coding accuracy.
public final class GemmaActivationWeightedScaleSearch {
    private let calibration: GemmaActivationStatistics
    private let development: GemmaActivationStatistics

    public init(calibration: GemmaActivationStatistics, development: GemmaActivationStatistics) throws {
        try calibration.requireDisjoint(from: development)
        try calibration.provenance.source.requireUnchanged()
        self.calibration = calibration
        self.development = development
    }

    public func rescore(
        modulePath: String, sourceWeight: MLXArray, templateWeight: MLXArray,
        templateScales: MLXArray, templateBiases: MLXArray, bits: Int, groupSize: Int
    ) throws -> GemmaActivationWeightedScaleSearchResult {
        guard isDecoderProjection(modulePath), let fit = calibration.moments[modulePath],
            let dev = development.moments[modulePath], fit.shape == dev.shape,
            sourceWeight.dtype == .bfloat16, [2, 3].contains(sourceWeight.ndim),
            sourceWeight.shape.allSatisfy({ $0 > 0 }), groupSize == 64, [4, 8].contains(bits),
            sourceWeight.dim(-1) % groupSize == 0,
            templateWeight.dtype == .uint32,
            templateScales.dtype == .bfloat16, templateBiases.dtype == .bfloat16
        else {
            throw GemmaActivationStatisticsError.invalid(
                "unsupported decoder projection, BF16 source or affine template")
        }
        let width = sourceWeight.dim(-1)
        let leading = Array(sourceWeight.shape.dropLast())
        guard templateWeight.shape == leading + [width * bits / 32],
            templateScales.shape == leading + [width / groupSize], templateBiases.shape == templateScales.shape,
            fit.shape == (sourceWeight.ndim == 2 ? [width] : [sourceWeight.dim(0), width])
        else { throw GemmaActivationStatisticsError.invalid("projection/template/statistics geometry mismatch") }
        try MLX.checkedEval(sourceWeight, templateScales, templateBiases)
        guard MLX.all(MLX.isFinite(sourceWeight)).item(Bool.self) else {
            throw GemmaActivationStatisticsError.invalid("non-finite BF16 source: \(modulePath)")
        }
        // LS2 joint affine fitting permits signed slopes. Their sign is part
        // of the stored grid; reject non-finite metadata, not valid orientation.
        guard MLX.all(MLX.isFinite(templateScales)).item(Bool.self) else {
            throw GemmaActivationStatisticsError.invalid("non-finite template scales: \(modulePath)")
        }
        guard MLX.all(MLX.isFinite(templateBiases)).item(Bool.self) else {
            throw GemmaActivationStatisticsError.invalid("non-finite template biases: \(modulePath)")
        }
        func retained(_ reason: String) -> GemmaActivationWeightedScaleSearchResult {
            GemmaActivationWeightedScaleSearchResult(
                weight: templateWeight, scales: templateScales, biases: templateBiases,
                diagnostics: [:], retainedTemplateExperts: sourceWeight.ndim == 3 ? Array(0..<sourceWeight.dim(0)) : [],
                retainedModuleReason: reason)
        }
        // Router decisions change which expert function runs. They are never
        // refined by this projection-error objective, regardless of precision.
        if modulePath.hasSuffix(".router.proj") { return retained("protected_router") }
        if bits == 8 { return retained("protected_q8_precision") }
        let result: GemmaActivationWeightedScaleSearchResult
        if sourceWeight.ndim == 3 {
            guard let fitCounts = calibration.expertCounts[modulePath],
                let devCounts = development.expertCounts[modulePath]
            else { throw GemmaActivationStatisticsError.invalid("missing routed expert counts") }
            let refined = try MistralActivationWeightedScaleSearch.rescoreExperts(
                sourceWeight: sourceWeight, templateWeight: templateWeight,
                templateScales: templateScales, templateBiases: templateBiases,
                calibrationSecondMoments: fit, calibrationPositionCounts: fitCounts,
                validationSecondMoments: dev, validationPositionCounts: devCounts,
                minimumExpertPositions: calibration.minimumExpertPositions)
            result = GemmaActivationWeightedScaleSearchResult(
                weight: refined.weight, scales: refined.scales, biases: refined.biases,
                diagnostics: refined.diagnostics, retainedTemplateExperts: refined.retainedTemplateExperts,
                retainedModuleReason: nil)
        } else {
            for moments in [fit, dev] {
                let values = moments.asArray(Float.self)
                for start in stride(from: 0, to: width, by: groupSize) {
                    guard values[start..<(start + groupSize)].contains(where: { $0 > 0 }) else {
                        return retained("silent_input_group")
                    }
                }
            }
            let refined = try MistralActivationWeightedScaleSearch.rescore(
                sourceWeight: sourceWeight, templateWeight: templateWeight,
                templateScales: templateScales, templateBiases: templateBiases,
                calibrationSecondMoments: fit, validationSecondMoments: dev)
            result = GemmaActivationWeightedScaleSearchResult(
                weight: refined.weight, scales: refined.scales, biases: refined.biases,
                diagnostics: [0: refined.diagnostics], retainedTemplateExperts: [], retainedModuleReason: nil)
        }
        for (actual, baseline) in [
            (result.weight, templateWeight), (result.scales, templateScales), (result.biases, templateBiases),
        ] {
            guard actual.shape == baseline.shape, actual.dtype == baseline.dtype, actual.nbytes == baseline.nbytes
            else {
                throw GemmaActivationStatisticsError.invalid("refinement changed stored template geometry")
            }
        }
        return result
    }

    private func isDecoderProjection(_ path: String) -> Bool {
        let roots = ["model.layers.", "language_model.model.layers."]
        guard let root = roots.first(where: { path.hasPrefix($0) }) else { return false }
        let suffix = path.dropFirst(root.count)
        guard let dot = suffix.firstIndex(of: "."), let layer = Int(suffix[..<dot]), layer >= 0 else { return false }
        let projection = String(suffix[suffix.index(after: dot)...])
        return [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj",
            "experts.switch_glu.gate_proj", "experts.switch_glu.up_proj", "experts.switch_glu.down_proj",
        ].contains(projection)
    }
}

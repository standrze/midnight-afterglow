// Extracted for Wick from Midnight: Sources/ModelRunnerCore/LagunaModel.swift
// Source snapshot SHA256: ade2066e22aadc016a89fe799e6dc0fada8a8d82201f2831ea52dfbc230b2e0a
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Native MLX-Swift implementation of Poolside's Laguna causal language model.
//
// Laguna XS.2 alternates full and sliding-window attention, varies the number
// of query heads per layer, applies QK normalization and per-head output gates,
// and uses a dense first MLP followed by top-k routed SwitchGLU MoE blocks.
// Keeping the implementation in ModelRunnerCore lets the runner support the
// architecture without carrying a private fork of mlx-swift-lm.

/// Task-scoped Laguna execution controls used by the native benchmark harness.
///
/// Child generation tasks inherit the value, while ordinary midnight calls use the optimized path by default.
public enum LagunaRuntimeTuning {
    @TaskLocal public static var useCompiledAttentionGate = true
    @TaskLocal public static var useCompiledMoEFusion = true
    // `nil` lets LocalModelRunner select the Metal-only production default.
    // Tests and the same-loaded benchmark use explicit values as scoped A/B
    // overrides, including an explicit false control.
    @TaskLocal public static var useCompiledBlockTail: Bool? = nil
    @TaskLocal public static var useFusedRouterTopK: Bool? = nil
    // Experimental affine Q4 gate/up + SiLU fusion. nil permits the runner's
    // explicit environment opt-in; ordinary execution continues to default off.
    @TaskLocal public static var useFusedGateUpSilu: Bool? = nil
}

enum LagunaCompiledBlockTailEligibility {
    static func allows(
        runtimeEnabled: Bool,
        capturesHiddenStates: Bool,
        batchSize: Int,
        sequenceLength: Int,
        cacheOffset: Int?,
        useCompiledAttentionGate: Bool,
        useCompiledMoEFusion: Bool
    ) -> Bool {
        runtimeEnabled
            && !capturesHiddenStates
            && batchSize == 1
            && sequenceLength == 1
            && (cacheOffset ?? 0) > 0
            && useCompiledAttentionGate
            && useCompiledMoEFusion
    }
}

public enum LagunaConfigurationError: Error, Equatable, LocalizedError {
    case invalidValue(String)
    case invalidLayerCount(field: String, expected: Int, actual: Int)
    case unsupportedValue(field: String, value: String)

    public var errorDescription: String? {
        switch self {
        case .invalidValue(let message):
            return "Invalid Laguna configuration: \(message)"
        case .invalidLayerCount(let field, let expected, let actual):
            return "Invalid Laguna configuration: \(field) has \(actual) entries; expected \(expected)."
        case .unsupportedValue(let field, let value):
            return "Unsupported Laguna configuration: \(field)=\(value)."
        }
    }
}

enum LagunaAttentionGating: Sendable {
    case disabled
    case perHead

    init(from container: KeyedDecodingContainer<LagunaConfiguration.CodingKeys>) throws {
        guard container.contains(.gating) else {
            self = .perHead
            return
        }
        if let enabled = try? container.decode(Bool.self, forKey: .gating) {
            self = enabled ? .perHead : .disabled
            return
        }

        let value = try container.decode(String.self, forKey: .gating)
        switch value.lowercased().replacingOccurrences(of: "_", with: "-") {
        case "per-head", "true": self = .perHead
        case "none", "false", "disabled": self = .disabled
        default: throw LagunaConfigurationError.unsupportedValue(field: "gating", value: value)
        }
    }
}

public struct LagunaConfiguration: Decodable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var attentionHeadsPerLayer: [Int]
    var keyValueHeads: Int
    var headDimension: Int
    var maxPositionEmbeddings: Int
    var rmsNormEpsilon: Float
    var attentionBias: Bool
    var qkvBias: Bool
    var gating: LagunaAttentionGating
    var tieWordEmbeddings: Bool
    var slidingWindow: Int?
    var partialRotaryFactor: Float?
    var ropeTheta: Float
    var ropeParameters: [String: [String: StringOrNumber]]?
    var layerTypes: [String]
    var mlpLayerTypes: [String]

    var numberOfExperts: Int
    var expertsPerToken: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var routedScalingFactor: Float
    var routerLogitSoftcap: Float
    var routerScoreFunction: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case attentionHeadsPerLayer = "num_attention_heads_per_layer"
        case keyValueHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEpsilon = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case qkvBias = "qkv_bias"
        case gating
        case tieWordEmbeddings = "tie_word_embeddings"
        case slidingWindow = "sliding_window"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case numberOfExperts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case routedScalingFactor = "moe_routed_scaling_factor"
        case routerLogitSoftcap = "moe_router_logit_softcapping"
        case routerScoreFunction = "moe_router_score_func"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "laguna"
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 100_352
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8_192
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 48
        keyValueHeads = try container.decodeIfPresent(Int.self, forKey: .keyValueHeads) ?? 8
        headDimension = try container.decodeIfPresent(Int.self, forKey: .headDimension) ?? 128
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        rmsNormEpsilon =
            try container.decodeIfPresent(Float.self, forKey: .rmsNormEpsilon) ?? 1e-6
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        qkvBias = try container.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
        gating = try LagunaAttentionGating(from: container)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        partialRotaryFactor = try container.decodeIfPresent(
            Float.self, forKey: .partialRotaryFactor)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeParameters = try container.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        layerTypes =
            try container.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        mlpLayerTypes =
            try container.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
            ?? ["dense"] + Array(repeating: "sparse", count: max(0, hiddenLayers - 1))
        attentionHeadsPerLayer =
            try container.decodeIfPresent(
                [Int].self, forKey: .attentionHeadsPerLayer)
            ?? Array(repeating: attentionHeads, count: hiddenLayers)

        numberOfExperts =
            try container.decodeIfPresent(Int.self, forKey: .numberOfExperts) ?? 256
        expertsPerToken =
            try container.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 8
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        sharedExpertIntermediateSize =
            try container.decodeIfPresent(
                Int.self, forKey: .sharedExpertIntermediateSize) ?? 512
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        routerLogitSoftcap =
            try container.decodeIfPresent(Float.self, forKey: .routerLogitSoftcap) ?? 0
        routerScoreFunction =
            try container.decodeIfPresent(String.self, forKey: .routerScoreFunction) ?? "sigmoid"

        try validate()
    }

    private func validate() throws {
        guard modelType == "laguna" else {
            throw LagunaConfigurationError.unsupportedValue(field: "model_type", value: modelType)
        }
        guard vocabularySize > 0, hiddenSize > 0, intermediateSize > 0,
            hiddenLayers > 0, attentionHeads > 0, keyValueHeads > 0,
            headDimension > 0, maxPositionEmbeddings > 0
        else {
            throw LagunaConfigurationError.invalidValue("model dimensions must all be positive")
        }
        guard rmsNormEpsilon > 0 else {
            throw LagunaConfigurationError.invalidValue("rms_norm_eps must be positive")
        }
        try Self.validateCount(layerTypes, field: "layer_types", expected: hiddenLayers)
        try Self.validateCount(mlpLayerTypes, field: "mlp_layer_types", expected: hiddenLayers)
        try Self.validateCount(
            attentionHeadsPerLayer,
            field: "num_attention_heads_per_layer",
            expected: hiddenLayers
        )

        for layerType in layerTypes
        where layerType != "full_attention"
            && layerType != "sliding_attention"
        {
            throw LagunaConfigurationError.unsupportedValue(
                field: "layer_types", value: layerType)
        }
        for mlpType in mlpLayerTypes where mlpType != "dense" && mlpType != "sparse" {
            throw LagunaConfigurationError.unsupportedValue(
                field: "mlp_layer_types", value: mlpType)
        }
        for heads in attentionHeadsPerLayer {
            guard heads > 0, heads.isMultiple(of: keyValueHeads) else {
                throw LagunaConfigurationError.invalidValue(
                    "every query-head count must be positive and divisible by num_key_value_heads"
                )
            }
        }
        if layerTypes.contains("sliding_attention") {
            guard let slidingWindow, slidingWindow > 0 else {
                throw LagunaConfigurationError.invalidValue(
                    "sliding_window must be positive when sliding attention is used")
            }
        }
        guard numberOfExperts > 0, expertsPerToken > 0,
            expertsPerToken <= numberOfExperts, moeIntermediateSize > 0,
            sharedExpertIntermediateSize > 0, routedScalingFactor > 0,
            routerLogitSoftcap >= 0
        else {
            throw LagunaConfigurationError.invalidValue("invalid MoE dimensions or routing values")
        }
        guard routerScoreFunction == "sigmoid" || routerScoreFunction == "sqrtsoftplus" else {
            throw LagunaConfigurationError.unsupportedValue(
                field: "moe_router_score_func", value: routerScoreFunction)
        }

        for layerType in Set(layerTypes) {
            let parameters = ropeParameters?[layerType]
            let partial =
                parameters?["partial_rotary_factor"]?.asFloat()
                ?? partialRotaryFactor ?? 1
            let rotaryDimensions = Int(Float(headDimension) * partial)
            guard partial > 0, partial <= 1, rotaryDimensions > 0,
                rotaryDimensions <= headDimension, rotaryDimensions.isMultiple(of: 2)
            else {
                throw LagunaConfigurationError.invalidValue(
                    "the rotary dimensions for \(layerType) must be positive, even, and no larger than head_dim"
                )
            }
        }
    }

    private static func validateCount<T>(
        _ values: [T], field: String, expected: Int
    ) throws {
        guard values.count == expected else {
            throw LagunaConfigurationError.invalidLayerCount(
                field: field, expected: expected, actual: values.count)
        }
    }

    func rope(for layerType: String) -> RoPELayer {
        let parameters = ropeParameters?[layerType] ?? [:]
        let base = parameters["rope_theta"]?.asFloat() ?? ropeTheta
        let partial =
            parameters["partial_rotary_factor"]?.asFloat()
            ?? partialRotaryFactor ?? 1
        return initializeRope(
            dims: Int(Float(headDimension) * partial),
            base: base,
            traditional: false,
            scalingConfig: parameters,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
    }
}

private let compiledLagunaSigmoidTopK8Router: @Sendable (MLXArray, MLXArray) -> (MLXArray, MLXArray) = compile {
    gates, correctionBias in
    lagunaSigmoidTopK8Router(gates, correctionBias: correctionBias)
}

// Laguna routes with biased FP32 sigmoid scores, but weights the selected
// experts with the corresponding unbiased scores. The generic chain performs
// a full stable sort plus gather and reduction for every sparse layer. Decode
// has exactly one 256-expert row, so one threadgroup can rank all experts and
// normalize the eight winners without those encoder-wide barriers.
private let lagunaRouterTopKSource = """
    uint row = threadgroup_position_in_grid.y;
    uint t = thread_position_in_threadgroup.x;

    threadgroup ulong ranking_keys[E_];
    threadgroup float top_scores[K_];

    float score = scores[row * E_ + t];
    float selection_score = score + correction_bias[t];
    uint bits = (selection_score == 0.0f) ? 0u : as_type<uint>(selection_score);
    uint monotone = isnan(selection_score)
        ? 0xFFFFFFFFu
        : (bits ^ ((uint)(((int)bits) >> 31) | 0x80000000u));

    // The legacy path sorts `-(score + bias)` ascending and is stable. Rank
    // larger selection scores first; for ties, rank the lower expert index
    // first. Inverting the index in the low bits supplies that tie-break.
    ulong key = (((ulong)monotone) << 32) | (ulong)(0xFFFFFFFFu - t);
    ranking_keys[t] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int above = 0;
    for (uint j = 0; j < E_; ++j) {
        above += (ranking_keys[j] > key) ? 1 : 0;
    }
    if (above < K_) {
        top_scores[above] = score;
        indices[row * K_ + above] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t == 0) {
        float total = 0.0f;
        for (int q = 0; q < K_; ++q) {
            total = top_scores[q] + total;
        }
        total = total + 1e-20f;
        for (int q = 0; q < K_; ++q) {
            weights[row * K_ + q] = static_cast<T>(top_scores[q] / total);
        }
    }
    """

private final class LagunaRouterTopKKernel: Sendable {
    static let shared = LagunaRouterTopKKernel()
    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "laguna_router_topk8_norm",
            inputNames: ["scores", "correction_bias"],
            outputNames: ["indices", "weights"],
            source: lagunaRouterTopKSource)
    }
}

func lagunaSigmoidTopK8Router(
    _ gates: MLXArray,
    correctionBias: MLXArray
) -> (weights: MLXArray, indices: MLXArray) {
    let scores = sigmoid(gates.asType(.float32))
    let indices = argPartition(
        -(scores + correctionBias), kth: 7, axis: -1
    )[.ellipsis, ..<8]
    var weights = takeAlong(scores, indices, axis: -1)
    weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
    return (weights.asType(gates.dtype), indices)
}

func fusedLagunaSigmoidTopK8Router(
    _ gates: MLXArray,
    correctionBias: MLXArray
) -> (weights: MLXArray, indices: MLXArray) {
    let scores = sigmoid(gates.asType(.float32))
    let expertCount = scores.dim(-1)
    precondition(expertCount == 256)
    precondition(scores.size == expertCount)
    precondition(correctionBias.size == expertCount)
    let shape = Array(scores.shape.dropLast()) + [8]
    let outputs = LagunaRouterTopKKernel.shared.kernel(
        [scores, correctionBias],
        template: [("T", gates.dtype), ("E_", expertCount), ("K_", 8)],
        grid: (expertCount, 1, 1),
        threadGroup: (expertCount, 1, 1),
        outputShapes: [shape, shape],
        outputDTypes: [.uint32, gates.dtype])
    return (outputs[1], outputs[0])
}

private let compiledLagunaMoEWeightedSharedResidual: @Sendable ([MLXArray]) -> [MLXArray] = compile(
    shapeless: true
) { inputs in
    let expertOutput = inputs[0]
    let weights = inputs[1].asType(expertOutput.dtype)
    let weighted = (expertOutput * expandedDimensions(weights, axis: -1)).sum(axis: -2)
    let scaled = weighted * inputs[2].asType(weighted.dtype)
    return [(scaled + inputs[3]) + inputs[4]]
}

// Laguna alternates full- and sliding-attention head counts. Keep this
// shape-specialized so MLX caches one valid graph for each head layout; the
// broadcast reshape is not safe in a single shapeless graph across both.
private let compiledLagunaPerHeadAttentionGate: @Sendable (MLXArray, MLXArray) -> MLXArray = compile {
    output, gateLogits in
    let gate = softplus(gateLogits.asType(.float32)).asType(output.dtype)
    return output * gate[.ellipsis, .newAxis]
}

private final class LagunaAttention: Module {
    let numberOfHeads: Int
    let numberOfKeyValueHeads: Int
    let headDimension: Int
    let scale: Float
    let usesSlidingWindow: Bool
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "g_proj") var gateProjection: Linear?

    init(_ configuration: LagunaConfiguration, layerIndex: Int) {
        numberOfHeads = configuration.attentionHeadsPerLayer[layerIndex]
        numberOfKeyValueHeads = configuration.keyValueHeads
        headDimension = configuration.headDimension
        scale = pow(Float(configuration.headDimension), -0.5)
        let layerType = configuration.layerTypes[layerIndex]
        usesSlidingWindow = layerType == "sliding_attention"
        rope = configuration.rope(for: layerType)

        _queryProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            numberOfHeads * headDimension,
            bias: configuration.qkvBias
        )
        _keyProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            numberOfKeyValueHeads * headDimension,
            bias: configuration.qkvBias
        )
        _valueProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            numberOfKeyValueHeads * headDimension,
            bias: configuration.qkvBias
        )
        _outputProjection.wrappedValue = Linear(
            numberOfHeads * headDimension,
            configuration.hiddenSize,
            bias: configuration.attentionBias
        )
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: headDimension, eps: configuration.rmsNormEpsilon)
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: headDimension, eps: configuration.rmsNormEpsilon)
        if configuration.gating == .perHead {
            _gateProjection.wrappedValue = Linear(
                configuration.hiddenSize, numberOfHeads, bias: false)
        }

        super.init()
    }

    func core(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var queries = queryProjection(x).reshaped(
            batch, length, numberOfHeads, headDimension)
        var keys = keyProjection(x).reshaped(
            batch, length, numberOfKeyValueHeads, headDimension)
        let values = valueProjection(x).reshaped(
            batch, length, numberOfKeyValueHeads, headDimension)

        queries = queryNorm(queries).transposed(0, 2, 1, 3)
        keys = keyNorm(keys).transposed(0, 2, 1, 3)
        let transposedValues = values.transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        return attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: transposedValues,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
    }

    func finish(
        _ perHeadOutput: MLXArray,
        normalizedInput x: MLXArray,
        useCompiledAttentionGate: Bool
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var output = perHeadOutput.reshaped(
            batch, length, numberOfHeads * headDimension)

        if let gateProjection {
            let gateLogits = gateProjection(x)
            let perHeadOutput = output.reshaped(batch, length, numberOfHeads, headDimension)
            if useCompiledAttentionGate {
                output = compiledLagunaPerHeadAttentionGate(perHeadOutput, gateLogits).reshaped(
                    batch, length, numberOfHeads * headDimension)
            } else {
                let gate = softplus(gateLogits.asType(.float32)).asType(output.dtype)
                output = (perHeadOutput * gate[.ellipsis, .newAxis]).reshaped(
                    batch, length, numberOfHeads * headDimension)
            }
        }

        return outputProjection(output)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        finish(
            core(x, mask: mask, cache: cache),
            normalizedInput: x,
            useCompiledAttentionGate: LagunaRuntimeTuning.useCompiledAttentionGate)
    }
}

#if os(macOS)
    /// Keep gate/up views independent: the pinned MLX compiler can retain a multi-output Split's sibling cycle after a compiled model is released.
    ///
    /// These equal slices preserve the projection layout and activation math.
    func lagunaGateUpSlices(_ projected: MLXArray) -> [MLXArray] {
        precondition(projected.dim(-1) % 2 == 0, "Gate/up projection must have equal halves")
        let half = projected.dim(-1) / 2
        return [projected[.ellipsis, ..<half], projected[.ellipsis, half...]]
    }
#endif

private final class LagunaMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") var gateUpProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(inputDimensions: Int, hiddenDimensions: Int) {
        _gateUpProjection.wrappedValue = Linear(
            inputDimensions, 2 * hiddenDimensions, bias: false)
        _downProjection.wrappedValue = Linear(
            hiddenDimensions, inputDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = gateUpProjection(x)
        #if os(macOS)
            let parts = lagunaGateUpSlices(gateUp)
        #else
            let parts = MLX.split(gateUp, parts: 2, axis: -1)
        #endif
        return downProjection(compiledSiluProduct(parts[0], parts[1]))
    }
}

private final class LagunaMoEGate: Module {
    let topK: Int
    let softcap: Float
    let scoreFunction: String

    @ModuleInfo(key: "proj") var projection: Linear
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ configuration: LagunaConfiguration) {
        topK = configuration.expertsPerToken
        softcap = configuration.routerLogitSoftcap
        scoreFunction = configuration.routerScoreFunction
        _projection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.numberOfExperts, bias: false)
        _correctionBias.wrappedValue = MLXArray.zeros([configuration.numberOfExperts])
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        useCompiledFusion: Bool,
        useFusedRouterTopK: Bool = false
    ) -> (indices: MLXArray, weights: MLXArray) {
        let projected = projection(x)
        if useCompiledFusion, topK == 8, softcap == 0, scoreFunction == "sigmoid" {
            if useFusedRouterTopK, projected.dim(-1) == 256,
                projected.size == projected.dim(-1), correctionBias.size == 256
            {
                let (weights, indices) = fusedLagunaSigmoidTopK8Router(
                    projected, correctionBias: correctionBias)
                return (indices, weights)
            }
            let (weights, indices) = compiledLagunaSigmoidTopK8Router(
                projected, correctionBias)
            return (indices, weights)
        }

        var logits = projected.asType(.float32)
        if softcap > 0 {
            logits = tanh(logits / softcap) * softcap
        }

        let scores: MLXArray
        switch scoreFunction {
        case "sqrtsoftplus": scores = sqrt(softplus(logits))
        default: scores = sigmoid(logits)
        }

        let scoresForSelection = scores + correctionBias
        let indices = argPartition(
            -scoresForSelection, kth: topK - 1, axis: -1
        )[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        // Routing is intentionally evaluated in FP32, but retaining FP32 here
        // promotes every expert output during the weighted reduction. Cast only
        // the normalized top-k weights back to the model dtype, matching Poolside's
        // reference path while keeping the numerically sensitive router in FP32.
        return (indices, weights.asType(x.dtype))
    }
}

private final class LagunaMoE: Module, UnaryLayer {
    private(set) var fusedGateUpSiluTraceCount = 0
    let routedScalingFactor: Float
    private var calibrationObserver: LagunaRoutedActivationObserver?
    private var calibrationPath: String?
    private var calibrationGateUp: SwitchLinear?
    private var calibrationDown: SwitchLinear?
    private let expertCount: Int
    var observesCalibration: Bool { calibrationObserver != nil }
    var supportsFusedGateUpSilu: Bool { LagunaFusedGateUpSiluBinding(switchMLP) != nil }

    @ModuleInfo(key: "gate") var gate: LagunaMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: FusedGateUpSwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaMLP

    init(_ configuration: LagunaConfiguration) {
        routedScalingFactor = configuration.routedScalingFactor
        expertCount = configuration.numberOfExperts
        _gate.wrappedValue = LagunaMoEGate(configuration)
        _switchMLP.wrappedValue = FusedGateUpSwitchGLU(
            inputDims: configuration.hiddenSize,
            hiddenDims: configuration.moeIntermediateSize,
            numExperts: configuration.numberOfExperts
        )
        _sharedExpert.wrappedValue = LagunaMLP(
            inputDimensions: configuration.hiddenSize,
            hiddenDimensions: configuration.sharedExpertIntermediateSize
        )
        super.init()
    }

    func setCalibrationObserver(_ observer: LagunaRoutedActivationObserver?, path: String) throws {
        guard let observer else {
            calibrationObserver = nil
            calibrationPath = nil
            calibrationGateUp = nil
            calibrationDown = nil
            return
        }
        let leaves = Dictionary(uniqueKeysWithValues: switchMLP.leafModules().flattened())
        guard let gateUp = leaves["gate_up_proj"] as? SwitchLinear,
            let down = leaves["down_proj"] as? SwitchLinear,
            !(gateUp is Quantized), !(down is Quantized)
        else {
            throw LagunaActivationStatisticsError.invalidInput(
                "\(path) requires unquantized fused gate/up and down projections")
        }
        calibrationObserver = observer
        calibrationPath = path
        calibrationGateUp = gateUp
        calibrationDown = down
    }

    private func calibratedExpertOutput(
        _ input: MLXArray, _ indices: MLXArray, useFusedGateUpSilu: Bool = false
    ) -> MLXArray {
        guard let observer = calibrationObserver, let path = calibrationPath,
            let gateUp = calibrationGateUp, let down = calibrationDown
        else {
            // This branch is traced only in the distinct opt-in compiled closure.
            // Preserve the ordinary path for every unsupported parameter/input shape.
            if useFusedGateUpSilu, !training, calibrationObserver == nil,
                let binding = LagunaFusedGateUpSiluBinding(switchMLP),
                let output = binding(input, indices)
            {
                // Runs while building the distinct compiled graph, never per token in
                // the cached trace. Benchmarks use it to reject silent fallback runs.
                fusedGateUpSiluTraceCount += 1
                return output
            }
            return switchMLP(input, indices)
        }

        // Match the ordinary fused switch layer's sorted and unsorted schedules.
        var x = MLX.expandedDimensions(input, axes: [-2, -3])
        var ids = indices
        var inverseOrder = MLXArray()
        let sorted = indices.size >= 64
        if sorted { (x, ids, inverseOrder) = gatherSort(x: x, indices: indices) }
        observer.observeRoutedProjection(
            path: path + ".gate_up_proj", input: x, indices: ids, expertCount: expertCount)
        #if os(macOS)
            let parts = lagunaGateUpSlices(gateUp(x, ids, sortedIndices: sorted))
        #else
            let parts = MLX.split(gateUp(x, ids, sortedIndices: sorted), parts: 2, axis: -1)
        #endif
        let activated = compiledSiluProduct(parts[0], parts[1])
        observer.observeRoutedProjection(
            path: path + ".down_proj", input: activated, indices: ids, expertCount: expertCount)
        x = down(activated, ids, sortedIndices: sorted)
        if sorted { x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape) }
        return MLX.squeezed(x, axis: -2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let route = gate(x, useCompiledFusion: false)
        let expertOutput = calibratedExpertOutput(x, route.indices)
        let routedOutput = weightedExpertSum(expertOutput, route.weights)
        return routedOutput * routedScalingFactor + sharedExpert(x)
    }

    func callAsFunction(
        _ x: MLXArray,
        adding residual: MLXArray,
        useCompiledFusion: Bool,
        useFusedRouterTopK: Bool = false,
        useFusedGateUpSilu: Bool = false
    ) -> MLXArray {
        guard useCompiledFusion else {
            return residual + callAsFunction(x)
        }

        let route = gate(
            x,
            useCompiledFusion: true,
            useFusedRouterTopK: useFusedRouterTopK)
        let expertOutput = calibratedExpertOutput(
            x, route.indices, useFusedGateUpSilu: useFusedGateUpSilu)
        let sharedOutput = sharedExpert(x)
        let scale = MLXArray(routedScalingFactor).asType(expertOutput.dtype)
        return compiledLagunaMoEWeightedSharedResidual(
            [expertOutput, route.weights, scale, sharedOutput, residual]
        )[0]
    }
}

private final class LagunaTransformerBlock: Module {
    let usesSlidingWindow: Bool

    @ModuleInfo(key: "self_attn") var attention: LagunaAttention
    let mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    private lazy var compiledDecodeTail: @Sendable ([MLXArray]) -> [MLXArray] =
        compile { [unowned self] inputs in
            [
                self.eagerTail(
                    residual: inputs[0],
                    normalizedAttentionInput: inputs[1],
                    perHeadAttentionOutput: inputs[2],
                    useCompiledAttentionGate: true,
                    useCompiledMoEFusion: true,
                    useFusedRouterTopK: false)
            ]
        }

    // Keep a distinct trace for the candidate. Reading a TaskLocal from inside
    // one compiled body would freeze whichever value was present at first trace
    // and make the same-loaded A/B compare two copies of one arm.
    private lazy var fusedRouterCompiledDecodeTail: @Sendable ([MLXArray]) -> [MLXArray] =
        compile { [unowned self] inputs in
            [
                self.eagerTail(
                    residual: inputs[0],
                    normalizedAttentionInput: inputs[1],
                    perHeadAttentionOutput: inputs[2],
                    useCompiledAttentionGate: true,
                    useCompiledMoEFusion: true,
                    useFusedRouterTopK: true)
            ]
        }

    // Separate from both existing traces: toggling the experiment on the same
    // loaded model must never reuse a graph traced for the opposite A/B arm.
    private lazy var fusedGateUpCompiledDecodeTail: @Sendable ([MLXArray]) -> [MLXArray] =
        compile { [unowned self] inputs in
            [
                self.eagerTail(
                    residual: inputs[0], normalizedAttentionInput: inputs[1],
                    perHeadAttentionOutput: inputs[2], useCompiledAttentionGate: true,
                    useCompiledMoEFusion: true, useFusedRouterTopK: true,
                    useFusedGateUpSilu: true)
            ]
        }

    init(_ configuration: LagunaConfiguration, layerIndex: Int) {
        let attention = LagunaAttention(configuration, layerIndex: layerIndex)
        usesSlidingWindow = attention.usesSlidingWindow
        _attention.wrappedValue = attention
        if configuration.mlpLayerTypes[layerIndex] == "sparse" {
            mlp = LagunaMoE(configuration)
        } else {
            mlp = LagunaMLP(
                inputDimensions: configuration.hiddenSize,
                hiddenDimensions: configuration.intermediateSize
            )
        }
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        useCompiledMoEFusion: Bool,
        useCompiledAttentionGate: Bool,
        useCompiledBlockTail: Bool,
        useFusedRouterTopK: Bool,
        useFusedGateUpSilu: Bool = false,
        capturesHiddenStates: Bool
    ) -> MLXArray {
        // Snapshot the offset before attention mutates the cache. This keeps a
        // one-token prompt at offset zero on the existing prefill path.
        let useCompiledTail = LagunaCompiledBlockTailEligibility.allows(
            runtimeEnabled: useCompiledBlockTail,
            capturesHiddenStates: capturesHiddenStates,
            batchSize: x.dim(0),
            sequenceLength: x.dim(1),
            cacheOffset: cache?.offset,
            useCompiledAttentionGate: useCompiledAttentionGate,
            useCompiledMoEFusion: useCompiledMoEFusion)
        let normalizedInput = inputNorm(x)
        let perHeadOutput = attention.core(normalizedInput, mask: mask, cache: cache)
        if useCompiledTail {
            if LagunaFusedGateUpSiluEligibility.allows(
                runtimeEnabled: useFusedGateUpSilu, useCompiledTail: useCompiledTail,
                useFusedRouter: useFusedRouterTopK, training: training,
                hasCalibrationObserver: (mlp as? LagunaMoE)?.observesCalibration ?? false)
            {
                return fusedGateUpCompiledDecodeTail([x, normalizedInput, perHeadOutput])[0]
            }
            if useFusedRouterTopK {
                return fusedRouterCompiledDecodeTail([x, normalizedInput, perHeadOutput])[0]
            }
            return compiledDecodeTail([x, normalizedInput, perHeadOutput])[0]
        }
        return eagerTail(
            residual: x,
            normalizedAttentionInput: normalizedInput,
            perHeadAttentionOutput: perHeadOutput,
            useCompiledAttentionGate: useCompiledAttentionGate,
            useCompiledMoEFusion: useCompiledMoEFusion,
            useFusedRouterTopK: false)
    }

    private func eagerTail(
        residual x: MLXArray,
        normalizedAttentionInput: MLXArray,
        perHeadAttentionOutput: MLXArray,
        useCompiledAttentionGate: Bool,
        useCompiledMoEFusion: Bool,
        useFusedRouterTopK: Bool,
        useFusedGateUpSilu: Bool = false
    ) -> MLXArray {
        let attended =
            x
            + attention.finish(
                perHeadAttentionOutput,
                normalizedInput: normalizedAttentionInput,
                useCompiledAttentionGate: useCompiledAttentionGate)
        if let sparseMLP = mlp as? LagunaMoE {
            return sparseMLP(
                postAttentionNorm(attended),
                adding: attended,
                useCompiledFusion: useCompiledMoEFusion,
                useFusedRouterTopK: useFusedRouterTopK,
                useFusedGateUpSilu: useFusedGateUpSilu)
        }
        return attended + mlp(postAttentionNorm(attended))
    }
}

public final class LagunaModelInner: Module {
    fileprivate var hasRoutedActivationObserver = false
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [LagunaTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    private let fullAttentionLayerIndex: Int?
    private let slidingAttentionLayerIndex: Int?
    private let slidingWindow: Int?

    init(_ configuration: LagunaConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        layers = (0..<configuration.hiddenLayers).map {
            LagunaTransformerBlock(configuration, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        fullAttentionLayerIndex = configuration.layerTypes.firstIndex(of: "full_attention")
        slidingAttentionLayerIndex = configuration.layerTypes.firstIndex(of: "sliding_attention")
        slidingWindow = configuration.slidingWindow
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forward(inputs, cache: cache, captureLayerIDs: [], captureLimit: nil).hidden
    }

    func forward(
        _ inputs: MLXArray,
        cache: [KVCache]?,
        captureLayerIDs: Set<Int>,
        captureLimit: Int?
    ) -> (hidden: MLXArray, auxiliaryHidden: MLXArray?) {
        var hidden = embedTokens(inputs)
        let useCompiledMoEFusion = LagunaRuntimeTuning.useCompiledMoEFusion
        let useCompiledAttentionGate = LagunaRuntimeTuning.useCompiledAttentionGate
        let useCompiledBlockTail =
            (LagunaRuntimeTuning.useCompiledBlockTail ?? false)
            && !hasRoutedActivationObserver
        let useFusedRouterTopK = LagunaRuntimeTuning.useFusedRouterTopK ?? false
        let useFusedGateUpSilu = LagunaRuntimeTuning.useFusedGateUpSilu ?? false
        let capturesHiddenStates = !captureLayerIDs.isEmpty
        var captured = [MLXArray]()
        captured.reserveCapacity(captureLayerIDs.count)

        let fullMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let fullAttentionLayerIndex {
                createAttentionMask(h: hidden, cache: cache?[fullAttentionLayerIndex])
            } else {
                .none
            }
        let slidingMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let slidingAttentionLayerIndex {
                createAttentionMask(
                    h: hidden,
                    cache: cache?[slidingAttentionLayerIndex],
                    windowSize: slidingWindow
                )
            } else {
                .none
            }

        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                mask: layer.usesSlidingWindow ? slidingMask : fullMask,
                cache: cache?[index],
                useCompiledMoEFusion: useCompiledMoEFusion,
                useCompiledAttentionGate: useCompiledAttentionGate,
                useCompiledBlockTail: useCompiledBlockTail,
                useFusedRouterTopK: useFusedRouterTopK,
                useFusedGateUpSilu: useFusedGateUpSilu,
                capturesHiddenStates: capturesHiddenStates
            )
            if captureLayerIDs.contains(index) {
                let start = captureLimit.map { max(0, hidden.dim(1) - $0) } ?? 0
                captured.append(hidden[0..., start..., 0...])
            }
        }
        let auxiliaryHidden =
            captured.isEmpty ? nil : concatenated(captured, axis: -1)
        return (norm(hidden), auxiliaryHidden)
    }
}

private final class LagunaCausalLanguageModel: Module {
    @ModuleInfo(key: "model") var model: LagunaModelInner
    @ModuleInfo(key: "lm_head") var languageModelHead: Linear?

    private let tiesWordEmbeddings: Bool

    init(_ configuration: LagunaConfiguration) {
        tiesWordEmbeddings = configuration.tieWordEmbeddings
        _model.wrappedValue = LagunaModelInner(configuration)
        if !configuration.tieWordEmbeddings {
            _languageModelHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        return logits(hidden)
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        if let languageModelHead {
            return languageModelHead(hidden)
        }
        precondition(tiesWordEmbeddings)
        return model.embedTokens.asLinear(hidden)
    }
}

public final class LagunaModel: Module, LLMModel, KVCacheDimensionProvider {
    public let modelType: String
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let configuration: LagunaConfiguration
    private var dflashTargetLayerIDs = Set<Int>()
    private var dflashCaptureWindow: Int?
    @ModuleInfo(key: "language_model") fileprivate var languageModel: LagunaCausalLanguageModel

    public init(_ configuration: LagunaConfiguration) {
        self.configuration = configuration
        modelType = configuration.modelType
        vocabularySize = configuration.vocabularySize
        kvHeads = Array(
            repeating: configuration.keyValueHeads,
            count: configuration.hiddenLayers
        )
        _languageModel.wrappedValue = LagunaCausalLanguageModel(configuration)
        super.init()
    }

    /// Metadata-only capability counts for rejecting an unsupported benchmark
    /// checkpoint before a fallback-only A/B run can be reported as an experiment.
    public var fusedGateUpSiluSparseLayerCount: Int {
        languageModel.model.layers.filter { $0.mlp is LagunaMoE }.count
    }

    public var fusedGateUpSiluTraceCount: Int {
        languageModel.model.layers.reduce(0) { $0 + (($1.mlp as? LagunaMoE)?.fusedGateUpSiluTraceCount ?? 0) }
    }

    public var fusedGateUpSiluEligibleLayerCount: Int {
        #if os(macOS)
            guard configuration.hiddenSize == 2048, configuration.moeIntermediateSize == 512,
                configuration.numberOfExperts == 256, configuration.expertsPerToken == 8
            else { return 0 }
            return languageModel.model.layers.reduce(0) { count, layer in
                count + (((layer.mlp as? LagunaMoE)?.supportsFusedGateUpSilu ?? false) ? 1 : 0)
            }
        #else
            return 0
        #endif
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Opt-in expert calibration for an unquantized teacher.
    ///
    /// Call with nil to restore normal serving. Collection automatically bypasses compiled block tails so the observer runs for every segment, including cached decode.
    public func setRoutedActivationObserver(_ observer: LagunaRoutedActivationObserver?) throws {
        if observer != nil,
            leafModules().flattened().contains(where: { $0.1 is Quantized })
        {
            throw LagunaActivationStatisticsError.invalidInput("use the unquantized Laguna teacher")
        }
        languageModel.model.hasRoutedActivationObserver = observer != nil
        for (index, layer) in languageModel.model.layers.enumerated() {
            if let moe = layer.mlp as? LagunaMoE {
                try moe.setCalibrationObserver(
                    observer, path: "language_model.model.layers.\(index).mlp.switch_mlp")
            }
        }
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        let emitDFlashState = state?[mtpEmitFlagKey] ?? false
        guard emitDFlashState else {
            return LMOutput(logits: languageModel(input.tokens, cache: cache))
        }
        precondition(
            !dflashTargetLayerIDs.isEmpty,
            "Laguna target received a DFlash state request before DFlash was configured")

        let output = languageModel.model.forward(
            input.tokens,
            cache: cache,
            captureLayerIDs: dflashTargetLayerIDs,
            captureLimit: dflashCaptureWindow
        )
        guard let auxiliaryHidden = output.auxiliaryHidden else {
            preconditionFailure("Laguna did not capture its configured DFlash target layers")
        }
        var outputState = state ?? LMOutput.State()
        outputState[mtpLastHiddenStatesKey] = auxiliaryHidden
        // DFlash owns projected context K/V and does not share target K/V. Empty
        // dictionaries satisfy the generic MTP state contract without retaining
        // any of the target's much larger attention tensors.
        outputState[mtpSharedKVStatesKey] = [:]
        outputState[mtpSharedKVOffsetsKey] = [:]
        outputState[mtpSharedKVSourceIndicesKey] = [:]
        return LMOutput(
            logits: languageModel.logits(output.hidden),
            state: outputState
        )
    }

    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard state?[mtpEmitFlagKey] == true else {
            return try prepareNormally(input, cache: cache, state: state, prefill: prefill)
        }

        let text = input.text
        let total = text.tokens.size
        guard total > 0 else { return .tokens(text) }
        let result = try withPreparedCache(cache, lengths: text.sequenceLengths) {
            var promptFeatures: MLXArray?
            var finalHidden: MLXArray?

            func process(_ range: Range<Int>) {
                let output = languageModel.model.forward(
                    text[.newAxis, range].tokens,
                    cache: cache,
                    captureLayerIDs: dflashTargetLayerIDs,
                    captureLimit: dflashCaptureWindow
                )
                guard let auxiliaryHidden = output.auxiliaryHidden else {
                    preconditionFailure("Laguna did not capture DFlash prompt features")
                }
                let combined =
                    promptFeatures.map {
                        concatenated([$0, auxiliaryHidden], axis: 1)
                    } ?? auxiliaryHidden
                let start = max(0, combined.dim(1) - (dflashCaptureWindow ?? combined.dim(1)))
                promptFeatures = combined[0..., start..., 0...]
                finalHidden = output.hidden[0..., (-1)..., 0...]
                asyncEval(cache, promptFeatures!, finalHidden!)
            }

            let processed = try prefill.forEachChunk(total: total, reserving: 0, process)
            if processed == 0 {
                process(0..<total)
                prefill.progress?(total, total)
            }
            guard let promptFeatures, let finalHidden else {
                preconditionFailure("Laguna DFlash prefill produced no target state")
            }
            eval(cache, promptFeatures, finalHidden)

            var outputState = state ?? LMOutput.State()
            outputState[mtpLastHiddenStatesKey] = promptFeatures
            outputState[mtpSharedKVStatesKey] = [:]
            outputState[mtpSharedKVOffsetsKey] = [:]
            outputState[mtpSharedKVSourceIndicesKey] = [:]
            return LMOutput(
                logits: languageModel.logits(finalHidden),
                state: outputState
            )
        }
        return .logits(result)
    }

    private func prepareNormally(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let stepSize = prefill.resolvedStepSize()
        let text = input.text
        let total = text.tokens.size
        guard total > stepSize else { return .tokens(text) }

        var processed = 0
        try withPreparedCache(cache, lengths: text.sequenceLengths) {
            var currentState = state
            processed = try prefill.forEachChunk(
                total: total,
                reserving: prefill.chunking == .remainder ? stepSize : 1
            ) { range in
                let output = self(
                    text[.newAxis, range],
                    cache: cache.isEmpty ? nil : cache,
                    state: currentState
                )
                currentState = output.state
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
        }
        return .tokens(text[processed...])
    }

    func configureDFlash(_ descriptor: LagunaDFlashTargetDescriptor) throws {
        guard configuration.hiddenSize == descriptor.hiddenSize else {
            throw LagunaConfigurationError.invalidValue(
                "DFlash hidden size \(descriptor.hiddenSize) does not match target hidden size \(configuration.hiddenSize)"
            )
        }
        guard configuration.vocabularySize == descriptor.vocabularySize else {
            throw LagunaConfigurationError.invalidValue(
                "DFlash vocabulary \(descriptor.vocabularySize) does not match target vocabulary \(configuration.vocabularySize)"
            )
        }
        guard configuration.hiddenLayers == descriptor.numberOfTargetLayers else {
            throw LagunaConfigurationError.invalidValue(
                "DFlash expects \(descriptor.numberOfTargetLayers) target layers; target has \(configuration.hiddenLayers)"
            )
        }
        guard !descriptor.targetLayerIDs.isEmpty,
            descriptor.targetLayerIDs.allSatisfy({ configuration.mlpLayerTypes.indices.contains($0) })
        else {
            throw LagunaConfigurationError.invalidValue(
                "DFlash target layer IDs are outside the target decoder")
        }
        guard descriptor.captureWindow > 0 else {
            throw LagunaConfigurationError.invalidValue("DFlash capture window must be positive")
        }
        dflashTargetLayerIDs = Set(descriptor.targetLayerIDs)
        dflashCaptureWindow = descriptor.captureWindow
    }

    func dflashTokenEmbeddings(_ tokens: MLXArray) -> MLXArray {
        languageModel.model.embedTokens(tokens)
    }

    func dflashRawLogits(_ hidden: MLXArray) -> MLXArray {
        languageModel.logits(hidden)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Self.sanitizedWeights(weights, configuration: configuration)
    }

    fileprivate static func sanitizedWeights(
        _ weights: [String: MLXArray], configuration: LagunaConfiguration
    ) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (originalKey, value) in weights {
            guard
                let key = Self.normalizedWeightKey(
                    originalKey, tiedWordEmbeddings: configuration.tieWordEmbeddings)
            else { continue }
            sanitized[key] = value
        }

        // Poolside's BF16 checkpoint stores one tensor per expert, while MLX's
        // SwitchLinear consumes a single leading expert dimension. Stack those
        // tensors here so the native Swift conversion path can quantize the
        // original checkpoint directly. The gate/up rows are fused before
        // quantization, which also makes them one indivisible precision unit under
        // the Q4R8 policy.
        for layerIndex in configuration.mlpLayerTypes.indices
        where configuration.mlpLayerTypes[layerIndex] == "sparse" {
            let mlpPrefix = "language_model.model.layers.\(layerIndex).mlp"
            let routerWeight = "\(mlpPrefix).gate.weight"
            let routerProjection = "\(mlpPrefix).gate.proj.weight"
            if let weight = sanitized.removeValue(forKey: routerWeight) {
                sanitized[routerProjection] = weight
            }

            let correctionBias = "\(mlpPrefix).experts.e_score_correction_bias"
            let routedCorrectionBias = "\(mlpPrefix).gate.e_score_correction_bias"
            if let bias = sanitized.removeValue(forKey: correctionBias) {
                sanitized[routedCorrectionBias] = bias
            }

            let destinationPrefix = "\(mlpPrefix).switch_mlp"
            for suffix in ["weight", "scales", "biases"] {
                let gateKeys = (0..<configuration.numberOfExperts).map {
                    "\(mlpPrefix).experts.\($0).gate_proj.\(suffix)"
                }
                let upKeys = (0..<configuration.numberOfExperts).map {
                    "\(mlpPrefix).experts.\($0).up_proj.\(suffix)"
                }
                if gateKeys.allSatisfy({ sanitized[$0] != nil })
                    && upKeys.allSatisfy({ sanitized[$0] != nil })
                {
                    let experts = zip(gateKeys, upKeys).map { pair in
                        concatenated(
                            [
                                sanitized.removeValue(forKey: pair.0)!,
                                sanitized.removeValue(forKey: pair.1)!,
                            ],
                            axis: -2
                        )
                    }
                    sanitized["\(destinationPrefix).gate_up_proj.\(suffix)"] =
                        MLX.stacked(experts)
                }

                let downKeys = (0..<configuration.numberOfExperts).map {
                    "\(mlpPrefix).experts.\($0).down_proj.\(suffix)"
                }
                if downKeys.allSatisfy({ sanitized[$0] != nil }) {
                    let experts = downKeys.map { sanitized.removeValue(forKey: $0)! }
                    sanitized["\(destinationPrefix).down_proj.\(suffix)"] =
                        MLX.stacked(experts)
                }
            }
        }

        // Laguna checkpoints store gate and up as separate projections.
        // Concatenating their output rows and per-row quantization metadata is
        // exact. Fusing both routed and always-active MLPs removes one quantized
        // matrix dispatch from every MLP without changing the Q4R8 values.
        for layerIndex in configuration.mlpLayerTypes.indices
        where configuration.mlpLayerTypes[layerIndex] == "sparse" {
            let prefix =
                "language_model.model.layers.\(layerIndex).mlp.switch_mlp"
            Self.fuseGateUpProjectionWeights(in: &sanitized, prefix: prefix)
        }

        for layerIndex in configuration.mlpLayerTypes.indices {
            let mlpPrefix = "language_model.model.layers.\(layerIndex).mlp"
            let prefix =
                configuration.mlpLayerTypes[layerIndex] == "sparse"
                ? "\(mlpPrefix).shared_expert"
                : mlpPrefix
            Self.fuseGateUpProjectionWeights(in: &sanitized, prefix: prefix)
        }

        return sanitized
    }

    static func fuseGateUpProjectionWeights(
        in weights: inout [String: MLXArray], prefix: String
    ) {
        for suffix in ["weight", "scales", "biases"] {
            let gateKey = "\(prefix).gate_proj.\(suffix)"
            let upKey = "\(prefix).up_proj.\(suffix)"
            let fusedKey = "\(prefix).gate_up_proj.\(suffix)"
            if weights[fusedKey] != nil {
                weights.removeValue(forKey: gateKey)
                weights.removeValue(forKey: upKey)
                continue
            }
            guard let gate = weights[gateKey], let up = weights[upKey] else {
                continue
            }
            weights[fusedKey] = concatenated([gate, up], axis: -2)
            weights.removeValue(forKey: gateKey)
            weights.removeValue(forKey: upKey)
        }
    }

    static func normalizedWeightKey(
        _ originalKey: String, tiedWordEmbeddings: Bool
    ) -> String? {
        let key =
            originalKey.hasPrefix("language_model.")
            ? originalKey
            : "language_model.\(originalKey)"
        if key.contains("self_attn.rotary_emb.inv_freq") {
            return nil
        }
        if tiedWordEmbeddings, key.hasPrefix("language_model.lm_head.") {
            return nil
        }
        return key
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.model.layers.map { layer in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: layer.usesSlidingWindow
            )
        }
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

public enum LagunaModelRegistration {
    public static func register() async {
        await LLMTypeRegistry.shared.registerModelType(
            "laguna",
            creator: { data in
                let configuration = try JSONDecoder().decode(
                    LagunaConfiguration.self, from: data)
                return LagunaModel(configuration)
            }
        )
    }

    public static func isRegistered() async -> Bool {
        await LLMTypeRegistry.shared.contains("laguna")
    }
}

/// A single native Laguna transformer block for bounded offline teacher runs.
///
/// It owns only this block, and always starts each segment with an empty KV cache.
public final class LagunaCalibrationBlock {
    private let block: LagunaTransformerBlock
    private let configuration: LagunaConfiguration
    public let layerIndex: Int
    public var module: Module { block }
    public var modulePrefix: String { "language_model.model.layers.\(layerIndex)" }

    public init(configuration: LagunaConfiguration, layerIndex: Int, sourceWeights: [String: MLXArray]) throws {
        guard (0..<configuration.hiddenLayers).contains(layerIndex) else {
            throw LagunaActivationStatisticsError.invalidInput("invalid calibration layer index")
        }
        self.configuration = configuration
        self.layerIndex = layerIndex
        block = LagunaTransformerBlock(configuration, layerIndex: layerIndex)
        let prefix = "language_model.model.layers.\(layerIndex)."
        let sanitized = LagunaModel.sanitizedWeights(sourceWeights, configuration: configuration)
        var weights = [String: MLXArray]()
        for (key, value) in sanitized {
            guard key.hasPrefix(prefix) else {
                throw LagunaActivationStatisticsError.invalidInput(
                    "unrelated tensor supplied to single-layer calibration")
            }
            weights[String(key.dropFirst(prefix.count))] = value
        }
        try block.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        for (path, module) in block.leafModules().flattened() where module is SwitchLinear {
            guard !(module is Quantized),
                let weight = Dictionary(uniqueKeysWithValues: module.parameters().flattened())["weight"],
                weight.ndim == 3, weight.dtype == .bfloat16
            else {
                throw LagunaActivationStatisticsError.invalidInput(
                    "layerwise teacher \(path) requires rank-3 BF16 expert weights")
            }
        }
        block.train(false)
        try MLX.checkedEval(block.parameters().flattened().map(\.1))
    }

    public func setRoutedActivationObserver(_ observer: LagunaRoutedActivationObserver?) throws {
        if let moe = block.mlp as? LagunaMoE {
            try moe.setCalibrationObserver(observer, path: modulePrefix + ".mlp.switch_mlp")
        }
    }

    public func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let mask = createAttentionMask(
            h: hidden, cache: nil,
            windowSize: block.usesSlidingWindow ? configuration.slidingWindow : nil)
        return block(
            hidden, mask: mask, cache: nil,
            useCompiledMoEFusion: LagunaRuntimeTuning.useCompiledMoEFusion,
            useCompiledAttentionGate: LagunaRuntimeTuning.useCompiledAttentionGate,
            useCompiledBlockTail: false, useFusedRouterTopK: false, capturesHiddenStates: false)
    }
}

public extension LagunaConfiguration {
    var calibrationLayerCount: Int { hiddenLayers }
    var calibrationHiddenSize: Int { hiddenSize }
    var calibrationRMSNormEpsilon: Float { rmsNormEpsilon }
    var calibrationTiesWordEmbeddings: Bool { tieWordEmbeddings }
}

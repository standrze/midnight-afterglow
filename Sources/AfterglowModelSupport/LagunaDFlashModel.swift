// Extracted for Wick from Midnight: Sources/ModelRunnerCore/LagunaDFlashModel.swift
// Source snapshot SHA256: cd8a3b7e9451e07a0394d3c8fa3e72887fe1b1b792d009c4dd04d09aa1426520
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Native MLX-Swift implementation of Poolside's Laguna-XS-2.1 DFlash drafter.
//
// The checkpoint deliberately has neither a token embedding nor an lm_head. It
// borrows both from the target Laguna model, turns five tapped target hidden
// states into per-layer context K/V, and fills all mask positions in one
// five-layer forward pass. Verification and target-cache transactions are
// provided by MLXLMCommon's MTPSpeculativeTokenIterator.

enum LagunaDFlashConfigurationError: Error, Equatable, LocalizedError {
    case invalidValue(String)
    case unsupportedValue(field: String, value: String)

    var errorDescription: String? {
        switch self {
        case .invalidValue(let detail):
            "Invalid Laguna DFlash configuration: \(detail)"
        case .unsupportedValue(let field, let value):
            "Unsupported Laguna DFlash configuration: \(field)=\(value)."
        }
    }
}

struct LagunaDFlashConfiguration: Decodable, Sendable {
    struct Block: Decodable, Sendable {
        let blockSize: Int
        let maskTokenID: Int
        let numberOfTargetLayers: Int
        let targetLayerIDs: [Int]
        let causal: Bool

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case maskTokenID = "mask_token_id"
            case numberOfTargetLayers = "num_target_layers"
            case targetLayerIDs = "target_layer_ids"
            case causal
        }
    }

    let modelType: String
    let architectures: [String]
    let vocabularySize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let keyValueHeads: Int
    let headDimension: Int
    let maxPositionEmbeddings: Int
    let rmsNormEpsilon: Float
    let ropeTheta: Float
    let slidingWindow: Int
    let layerTypes: [String]
    let attentionBias: Bool
    let block: Block

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEpsilon = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case attentionBias = "attention_bias"
        case block = "dflash_config"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        keyValueHeads = try container.decode(Int.self, forKey: .keyValueHeads)
        headDimension = try container.decode(Int.self, forKey: .headDimension)
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        rmsNormEpsilon = try container.decode(Float.self, forKey: .rmsNormEpsilon)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        block = try container.decode(Block.self, forKey: .block)
        try validate()
    }

    private func validate() throws {
        guard modelType == "laguna" else {
            throw LagunaDFlashConfigurationError.unsupportedValue(
                field: "model_type", value: modelType)
        }
        guard architectures.contains("DFlashLagunaForCausalLM") else {
            throw LagunaDFlashConfigurationError.invalidValue(
                "architectures must contain DFlashLagunaForCausalLM")
        }
        guard vocabularySize > 0, hiddenSize > 0, intermediateSize > 0,
            hiddenLayers > 0, attentionHeads > 0, keyValueHeads > 0,
            headDimension > 0, maxPositionEmbeddings > 0, rmsNormEpsilon > 0,
            ropeTheta > 0, slidingWindow > 0
        else {
            throw LagunaDFlashConfigurationError.invalidValue(
                "model dimensions, norm epsilon, RoPE theta, and sliding window must be positive")
        }
        guard attentionHeads.isMultiple(of: keyValueHeads) else {
            throw LagunaDFlashConfigurationError.invalidValue(
                "num_attention_heads must be divisible by num_key_value_heads")
        }
        guard !attentionBias else {
            throw LagunaDFlashConfigurationError.unsupportedValue(
                field: "attention_bias", value: "true")
        }
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "sliding_attention" })
        else {
            throw LagunaDFlashConfigurationError.invalidValue(
                "every draft layer must be sliding_attention")
        }
        guard block.causal else {
            throw LagunaDFlashConfigurationError.unsupportedValue(
                field: "dflash_config.causal", value: "false")
        }
        guard block.blockSize >= 2 else {
            throw LagunaDFlashConfigurationError.invalidValue(
                "dflash_config.block_size must be at least 2")
        }
        guard block.maskTokenID >= 0, block.maskTokenID < vocabularySize else {
            throw LagunaDFlashConfigurationError.invalidValue(
                "dflash_config.mask_token_id is outside the vocabulary")
        }
        guard block.numberOfTargetLayers > 0 else {
            throw LagunaDFlashConfigurationError.invalidValue(
                "dflash_config.num_target_layers must be positive")
        }
        guard !block.targetLayerIDs.isEmpty,
            block.targetLayerIDs == block.targetLayerIDs.sorted(),
            Set(block.targetLayerIDs).count == block.targetLayerIDs.count,
            block.targetLayerIDs.allSatisfy({ $0 >= 0 && $0 < block.numberOfTargetLayers })
        else {
            throw LagunaDFlashConfigurationError.invalidValue(
                "dflash_config.target_layer_ids must be unique, ascending, and in range")
        }
    }
}

struct LagunaDFlashTargetDescriptor: Sendable, Equatable {
    let hiddenSize: Int
    let vocabularySize: Int
    let numberOfTargetLayers: Int
    let targetLayerIDs: [Int]
    let captureWindow: Int
    let blockSize: Int
}

final class LagunaDFlashAttention: Module {
    let numberOfHeads: Int
    let numberOfKeyValueHeads: Int
    let headDimension: Int
    let scale: Float
    let slidingWindow: Int
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "kv_proj") var keyValueProjection: Linear
    @ModuleInfo(key: "g_proj") var gateProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    init(_ configuration: LagunaDFlashConfiguration) {
        numberOfHeads = configuration.attentionHeads
        numberOfKeyValueHeads = configuration.keyValueHeads
        headDimension = configuration.headDimension
        scale = pow(Float(configuration.headDimension), -0.5)
        slidingWindow = configuration.slidingWindow
        rope = initializeRope(
            dims: configuration.headDimension,
            base: configuration.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: configuration.maxPositionEmbeddings
        )
        _queryProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.attentionHeads * configuration.headDimension,
            bias: false
        )
        _keyValueProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            2 * configuration.keyValueHeads * configuration.headDimension,
            bias: false
        )
        _gateProjection.wrappedValue = Linear(
            configuration.hiddenSize, configuration.attentionHeads, bias: false)
        _outputProjection.wrappedValue = Linear(
            configuration.attentionHeads * configuration.headDimension,
            configuration.hiddenSize,
            bias: false
        )
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension, eps: configuration.rmsNormEpsilon)
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: configuration.headDimension, eps: configuration.rmsNormEpsilon)
        super.init()
    }

    func appendContext(
        _ context: MLXArray,
        cache: KVCache,
        position: Int
    ) {
        let batch = context.dim(0)
        let length = context.dim(1)
        let kv = keyValueProjection(context)
        let parts = MLX.split(kv, parts: 2, axis: -1)
        var keys = parts[0].reshaped(
            batch, length, numberOfKeyValueHeads, headDimension)
        let values = parts[1].reshaped(
            batch, length, numberOfKeyValueHeads, headDimension)
        keys = keyNorm(keys).transposed(0, 2, 1, 3)
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(position))
        _ = cache.update(
            keys: keys,
            values: values.transposed(0, 2, 1, 3)
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: KVCache,
        position: Int
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var queries = queryProjection(x).reshaped(
            batch, length, numberOfHeads, headDimension)
        let kv = keyValueProjection(x)
        let parts = MLX.split(kv, parts: 2, axis: -1)
        var keys = parts[0].reshaped(
            batch, length, numberOfKeyValueHeads, headDimension)
        let values = parts[1].reshaped(
            batch, length, numberOfKeyValueHeads, headDimension)

        queries = queryNorm(queries).transposed(0, 2, 1, 3)
        keys = keyNorm(keys).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(position))
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(position))

        let mask = cache.makeMask(
            n: length, windowSize: slidingWindow, returnArray: false)
        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values.transposed(0, 2, 1, 3),
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)

        let gate = softplus(gateProjection(x).asType(.float32)).asType(output.dtype)
        output = output * gate[.ellipsis, .newAxis]
        return outputProjection(
            output.reshaped(batch, length, numberOfHeads * headDimension))
    }
}

final class LagunaDFlashMLP: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ configuration: LagunaDFlashConfiguration) {
        _gateUpProjection.wrappedValue = Linear(
            configuration.hiddenSize, 2 * configuration.intermediateSize, bias: false)
        _downProjection.wrappedValue = Linear(
            configuration.intermediateSize, configuration.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(gateUpProjection(x), parts: 2, axis: -1)
        return downProjection(MLXNN.silu(parts[0]) * parts[1])
    }
}

final class LagunaDFlashLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LagunaDFlashAttention
    @ModuleInfo(key: "mlp") var mlp: LagunaDFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(_ configuration: LagunaDFlashConfiguration) {
        _attention.wrappedValue = LagunaDFlashAttention(configuration)
        _mlp.wrappedValue = LagunaDFlashMLP(configuration)
        _inputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        _postAttentionNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        super.init()
    }

    func appendContext(_ context: MLXArray, cache: KVCache, position: Int) {
        // Laguna's DFlash checkpoint applies the layer input norm to context rows
        // before the K/V projection. This convention is not encoded in config.json.
        attention.appendContext(inputNorm(context), cache: cache, position: position)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache, position: Int) -> MLXArray {
        let attended = x + attention(inputNorm(x), cache: cache, position: position)
        return attended + mlp(postAttentionNorm(attended))
    }
}

final class LagunaDFlashModel: Module, StatefulMTPDrafterModel {
    let configuration: LagunaDFlashConfiguration
    let maximumBlockSize: Int?
    let requiresSharedTargetKV = false
    let requiresPromptPrefill = true
    let requiresGreedySampling = true

    // MLXLMCommon uses this to accept a bounded suffix of target prompt
    // features. Every DFlash layer is SWA, so earlier rows cannot affect a
    // proposal and retaining them would waste about 20 KiB per prompt token.
    var promptHiddenStateWindow: Int? { configuration.slidingWindow }
    let supportsChunkedPromptPrefill = true

    @ModuleInfo(key: "fc") var contextProjection: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "norm") var outputNorm: RMSNorm
    @ModuleInfo(key: "aux_hidden_norms") var auxiliaryHiddenNorms: [RMSNorm]
    @ModuleInfo(key: "layers") var layers: [LagunaDFlashLayer]

    init(_ configuration: LagunaDFlashConfiguration) {
        self.configuration = configuration
        maximumBlockSize = configuration.block.blockSize
        _contextProjection.wrappedValue = Linear(
            configuration.hiddenSize * configuration.block.targetLayerIDs.count,
            configuration.hiddenSize,
            bias: false
        )
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        _outputNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        _auxiliaryHiddenNorms.wrappedValue = configuration.block.targetLayerIDs.map { _ in
            RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.rmsNormEpsilon)
        }
        _layers.wrappedValue = (0..<configuration.hiddenLayers).map { _ in
            LagunaDFlashLayer(configuration)
        }
        super.init()
    }

    var targetDescriptor: LagunaDFlashTargetDescriptor {
        LagunaDFlashTargetDescriptor(
            hiddenSize: configuration.hiddenSize,
            vocabularySize: configuration.vocabularySize,
            numberOfTargetLayers: configuration.block.numberOfTargetLayers,
            targetLayerIDs: configuration.block.targetLayerIDs,
            captureWindow: configuration.slidingWindow,
            blockSize: configuration.block.blockSize
        )
    }

    func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(
            cache: layers.map { _ in RotatingKVCache(maxSize: configuration.slidingWindow) })
    }

    func prepareDrafterState(
        target _: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        let promptLength = promptTokens.dim(-1)
        let featureCount = targetHidden.dim(1)
        precondition(
            featureCount <= promptLength,
            "Laguna DFlash received more target features than prompt tokens")
        state.nextPosition = promptLength - featureCount
        appendTargetContext(targetHidden, state: &state)
        precondition(
            state.nextPosition == promptLength,
            "Laguna DFlash prompt context did not advance to the target position")
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        let hiddenLength = lastHidden.dim(1)
        precondition(
            queryOffset >= hiddenLength,
            "Laguna DFlash query offset precedes its fallback context")
        state.nextPosition = queryOffset - hiddenLength
        appendTargetContext(lastHidden, state: &state)
        return draftBlock(
            target: target,
            lastToken: lastToken,
            lastHidden: lastHidden,
            sharedKV: sharedKV,
            positionDeltas: positionDeltas,
            queryOffset: queryOffset,
            blockSize: blockSize,
            state: &state,
            sampler: sampler
        )
    }

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden _: MLXArray,
        sharedKV _: [String: (MLXArray, MLXArray)],
        positionDeltas _: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        precondition(blockSize >= 2 && blockSize <= configuration.block.blockSize)
        guard let target = target as? LagunaModel else {
            fatalError("Laguna DFlash requires a LagunaModel target, got \(type(of: target))")
        }
        precondition(
            queryOffset == state.nextPosition,
            "Laguna DFlash context is at \(state.nextPosition), target is at \(queryOffset)")

        let anchor =
            lastToken.ndim == 1
            ? lastToken.reshaped(lastToken.dim(0), 1)
            : lastToken
        let masks = MLXArray.full(
            [anchor.dim(0), blockSize - 1],
            values: MLXArray(Int32(configuration.block.maskTokenID)),
            dtype: .int32
        )
        let blockTokens = concatenated([anchor.asType(.int32), masks], axis: 1)
        var hidden = target.dflashTokenEmbeddings(blockTokens)

        // Proposal rows never become committed draft context. Run them against
        // independent cache copies so a rotating cache remains rollback-safe even
        // after the 512-token window has wrapped.
        let proposalCaches = state.cache.map { $0.copy() }
        for (layer, cache) in zip(layers, proposalCaches) {
            hidden = layer(hidden, cache: cache, position: queryOffset)
        }
        hidden = outputNorm(hidden)
        let logits = target.dflashRawLogits(hidden[0..., 1..., 0...])
        return sampler.sample(logits: logits)
    }

    func commitDrafterState(
        target _: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens _: MLXArray,
        acceptedCount: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        // The verifier input is [incoming bonus, draft_1, ...]. The bonus and
        // accepted draft prefix are committed; the sampled correction/next bonus
        // has not been evaluated by the target yet.
        let committedCount = acceptedCount + 1
        precondition(targetHidden.dim(1) >= committedCount)
        appendTargetContext(
            targetHidden[0..., ..<committedCount, 0...],
            state: &state
        )
    }

    private func appendTargetContext(
        _ targetHidden: MLXArray,
        state: inout MTPDrafterState
    ) {
        guard targetHidden.dim(1) > 0 else { return }
        precondition(
            targetHidden.dim(-1)
                == configuration.hiddenSize * configuration.block.targetLayerIDs.count,
            "Laguna DFlash target feature width mismatch")
        precondition(state.cache.count == layers.count)

        let slices = auxiliaryHiddenNorms.indices.map { index in
            let start = index * configuration.hiddenSize
            return auxiliaryHiddenNorms[index](
                targetHidden[0..., 0..., start..<(start + configuration.hiddenSize)])
        }
        let fused = hiddenNorm(contextProjection(concatenated(slices, axis: -1)))
        for (layer, cache) in zip(layers, state.cache) {
            layer.appendContext(fused, cache: cache, position: state.nextPosition)
        }
        state.nextPosition += targetHidden.dim(1)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        let queryRows = configuration.attentionHeads * configuration.headDimension

        for layerIndex in 0..<configuration.hiddenLayers {
            let attention = "layers.\(layerIndex).self_attn"
            for suffix in ["weight", "scales", "biases"] {
                let fusedKey = "\(attention).qkv_proj.\(suffix)"
                guard let fused = sanitized.removeValue(forKey: fusedKey) else { continue }
                sanitized["\(attention).q_proj.\(suffix)"] = fused[0..<queryRows, 0...]
                sanitized["\(attention).kv_proj.\(suffix)"] = fused[queryRows..., 0...]
            }

            let mlp = "layers.\(layerIndex).mlp"
            for suffix in ["weight", "scales", "biases"] {
                let gateKey = "\(mlp).gate_proj.\(suffix)"
                let upKey = "\(mlp).up_proj.\(suffix)"
                guard let gate = sanitized.removeValue(forKey: gateKey),
                    let up = sanitized.removeValue(forKey: upKey)
                else { continue }
                sanitized["\(mlp).gate_up_proj.\(suffix)"] =
                    concatenated([gate, up], axis: -2)
            }
        }
        return sanitized
    }
}

public enum LagunaDFlashRegistration {
    private actor Registrar {
        private var registered = false

        func register() async {
            guard !registered else { return }
            // Set this before crossing to the registry actor so concurrent runner
            // initializations cannot append the same predicate more than once.
            registered = true
            await MTPDrafterTypeRegistry.shared.registerModelType(
                "laguna",
                matches: { data in
                    guard
                        let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                        let architectures = object["architectures"] as? [String]
                    else { return false }
                    return architectures.contains("DFlashLagunaForCausalLM")
                },
                creator: { data in
                    LagunaDFlashModel(
                        try JSONDecoder.json5().decode(LagunaDFlashConfiguration.self, from: data))
                }
            )
        }
    }

    private static let registrar = Registrar()

    public static func register() async {
        await registrar.register()
    }

    public static func isRegistered() async -> Bool {
        await MTPDrafterTypeRegistry.shared.contains("laguna")
    }
}

// Extracted for Wick from Midnight: Sources/ModelRunnerCore/TalkieModel.swift
// Source snapshot SHA256: a197b468a4ca33df99b56906c09ead832f2c7f9f2821d7350d8b4396a80eb425
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

// Native Talkie (talkie-lm/talkie), with MLX cache and quantized-linear support.
// Equations cross-checked against the authors' model.py and Apple's mlx-lm
// talkie.py (173d49d288a61fee7bfe122d8002ff9cad2b8330, MIT).
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public enum TalkieConfigurationError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self {
        case .invalid(let detail): "Invalid Talkie configuration: \(detail)"
        }
    }
}

public struct TalkieConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDimension: Int
    public let intermediateSize: Int
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let rmsNormEpsilon: Float
    public let maxPositionEmbeddings: Int
    let permitsProjectionFusion: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDimension = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case rmsNormEpsilon = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
    }

    // Heterogeneous per-module quantization uses source module names. Keep those
    // modules separate so the generic loader can apply the exact declared recipe.
    private struct QuantizationShape: Decodable {
        struct Key: CodingKey {
            let stringValue: String
            let intValue: Int? = nil
            init?(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }
        let uniform: Bool
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: Key.self)
            let keys = c.allKeys.map(\.stringValue)
            let mode = try c.decodeIfPresent(String.self, forKey: Key(stringValue: "mode")!) ?? "affine"
            uniform = mode == "affine" && Set(keys).isSubset(of: ["group_size", "bits", "mode"])
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "talkie"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 40
        headDimension = try c.decodeIfPresent(Int.self, forKey: .headDimension) ?? 128
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 13696
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 65540
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        // PyTorch's weightless F.rms_norm defaults to fp32 machine epsilon.
        rmsNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .rmsNormEpsilon) ?? Float.ulpOfOne
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        keyValueHeads = try c.decodeIfPresent(Int.self, forKey: .keyValueHeads) ?? attentionHeads
        let tied = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        permitsProjectionFusion = try c.decodeIfPresent(QuantizationShape.self, forKey: .quantization)?.uniform ?? true
        guard modelType == "talkie", !tied else {
            throw TalkieConfigurationError.invalid("requires model_type=talkie and untied output weights")
        }
        guard (1...4096).contains(hiddenLayers), (1...65536).contains(hiddenSize),
            (1...4096).contains(attentionHeads), (2...4096).contains(headDimension),
            headDimension.isMultiple(of: 2), hiddenSize == attentionHeads * headDimension,
            (1...attentionHeads).contains(keyValueHeads), attentionHeads.isMultiple(of: keyValueHeads),
            (1...1_048_576).contains(intermediateSize),
            (1...1_048_576).contains(vocabularySize), (1...16_777_216).contains(maxPositionEmbeddings),
            ropeTheta.isFinite, ropeTheta > 0, rmsNormEpsilon.isFinite, rmsNormEpsilon > 0
        else {
            throw TalkieConfigurationError.invalid(
                "invalid dimensions, attention geometry, rotary base, or normalization epsilon")
        }
    }
}

@inline(__always)
private func talkieNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    // The Swift API's weight argument is nonoptional; mlxNone represents the
    // underlying kernel's optional weight, avoiding fake learned norm parameters.
    MLXFast.rmsNorm(x, weight: .mlxNone, eps: eps)
}

private final class TalkieActGain: Module {
    @ParameterInfo(key: "a_g") var gain: MLXArray
    init(_ initialValue: Float) {
        _gain.wrappedValue = MLXArray([initialValue])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gain.asType(x.dtype) }
}

private final class TalkieHeadGain: Module {
    @ParameterInfo(key: "head_g") var gain: MLXArray
    init(_ heads: Int) {
        _gain.wrappedValue = MLXArray.ones([heads])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gain.asType(x.dtype).reshaped(1, -1, 1, 1)
    }
}

private final class TalkieAttention: Module {
    @ModuleInfo(key: "attn_query") var query: Linear?
    @ModuleInfo(key: "attn_key") var key: Linear?
    @ModuleInfo(key: "attn_value") var value: Linear?
    @ModuleInfo(key: "attn_qkv") var qkv: Linear?
    @ModuleInfo(key: "attn_resid") var output: Linear
    @ModuleInfo(key: "head_gain") var headGain: TalkieHeadGain
    private let heads: Int
    private let kvHeads: Int
    private let headDimension: Int
    private let eps: Float
    private let scale: Float
    private let _frequencies: MLXArray

    init(_ c: TalkieConfiguration, fused: Bool) {
        heads = c.attentionHeads
        kvHeads = c.keyValueHeads
        headDimension = c.headDimension
        eps = c.rmsNormEpsilon
        scale = pow(Float(c.headDimension), -0.5)
        _frequencies = -pow(
            c.ropeTheta, MLXArray(stride(from: 0, to: c.headDimension, by: 2)).asType(.float32) / Float(c.headDimension)
        )
        if fused {
            _qkv.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: false)
        } else {
            _query.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
            _key.wrappedValue = Linear(c.hiddenSize, c.keyValueHeads * c.headDimension, bias: false)
            _value.wrappedValue = Linear(c.hiddenSize, c.keyValueHeads * c.headDimension, bias: false)
        }
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _headGain.wrappedValue = TalkieHeadGain(c.attentionHeads)
        super.init()
    }

    private func rope(_ x: MLXArray, offset: RoPEOffset?) -> MLXArray {
        // Talkie uses the inverse of NeoX's rotation. Negative frequency divisors
        // preserve the native fused RoPE kernel and both scalar and batched offsets.
        switch offset {
        case .batch(let offsets):
            return MLXFast.RoPE(
                x, dimensions: headDimension, traditional: false,
                base: nil, scale: 1, offset: offsets, freqs: _frequencies)
        case .scalar(let offset):
            return MLXFast.RoPE(
                x, dimensions: headDimension, traditional: false,
                base: nil, scale: 1, offset: offset, freqs: _frequencies)
        case nil:
            return MLXFast.RoPE(
                x, dimensions: headDimension, traditional: false,
                base: nil, scale: 1, offset: 0, freqs: _frequencies)
        }
    }

    private func projectKV(_ layer: Linear, _ x: MLXArray) -> MLXArray {
        if layer.weight.dtype == .float32 && x.dtype == .bfloat16 {
            // Recovery uses FP32 master parameters, but rounds them before the
            // forward GEMM so training matches the exported BF16 checkpoint.
            // K/V projections are constructed without biases.
            return matmul(x, layer.weight.asType(.bfloat16).T)
        }
        return layer(x)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let b = x.dim(0)
        let length = x.dim(1)
        var q: MLXArray
        var k: MLXArray
        var v: MLXArray
        if let qkv {
            let projected = qkv(x)
            let width = heads * headDimension
            q = projected[.ellipsis, 0..<width]
            k = projected[.ellipsis, width..<(2 * width)]
            v = projected[.ellipsis, (2 * width)...]
        } else {
            q = query!(x)
            k = projectKV(key!, x)
            v = projectKV(value!, x)
        }
        q = q.reshaped(b, length, heads, headDimension).transposed(0, 2, 1, 3)
        k = k.reshaped(b, length, kvHeads, headDimension).transposed(0, 2, 1, 3)
        v = v.reshaped(b, length, kvHeads, headDimension).transposed(0, 2, 1, 3)
        let offset = cache?.ropeOffset
        q = headGain(talkieNorm(rope(q, offset: offset), eps: eps))
        k = talkieNorm(rope(k, offset: offset), eps: eps)
        let attended = attentionWithCacheUpdate(
            queries: q, keys: k, values: v,
            cache: cache, scale: scale, mask: mask)
        return output(attended.transposed(0, 2, 1, 3).reshaped(b, length, -1))
    }
}

private final class TalkieMLP: Module {
    @ModuleInfo(key: "mlp_gate") var gate: Linear?
    @ModuleInfo(key: "mlp_linear") var up: Linear?
    @ModuleInfo(key: "mlp_gate_up") var gateUp: Linear?
    @ModuleInfo(key: "mlp_resid") var down: Linear
    private let intermediateSize: Int
    init(_ c: TalkieConfiguration, fused: Bool) {
        intermediateSize = c.intermediateSize
        if fused {
            _gateUp.wrappedValue = Linear(c.hiddenSize, 2 * c.intermediateSize, bias: false)
        } else {
            _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
            _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        }
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let gateUp {
            let pair = gateUp(x)
            return down(silu(pair[.ellipsis, 0..<intermediateSize]) * pair[.ellipsis, intermediateSize...])
        }
        return down(silu(gate!(x)) * up!(x))
    }
}

private final class TalkieBlock: Module {
    let attn: TalkieAttention
    let mlp: TalkieMLP
    @ModuleInfo(key: "attn_gain") var attentionGain: TalkieActGain
    @ModuleInfo(key: "mlp_gain") var mlpGain: TalkieActGain
    @ModuleInfo(key: "embed_skip") var embeddingSkip: TalkieActGain
    private let eps: Float
    init(_ c: TalkieConfiguration, fused: Bool) {
        attn = TalkieAttention(c, fused: fused)
        mlp = TalkieMLP(c, fused: fused)
        let gain = pow(Float(2 * c.hiddenLayers), -0.5)
        _attentionGain.wrappedValue = TalkieActGain(gain)
        _mlpGain.wrappedValue = TalkieActGain(gain)
        _embeddingSkip.wrappedValue = TalkieActGain(0)
        eps = c.rmsNormEpsilon
        super.init()
    }
    func callAsFunction(
        _ embedding: MLXArray, _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attentionGain(attn(talkieNorm(x, eps: eps), mask: mask, cache: cache))
        let out = h + mlpGain(mlp(talkieNorm(h, eps: eps)))
        return out + embeddingSkip(embedding)
    }
}

private final class TalkieInner: Module {
    @ModuleInfo var embed: Embedding
    let blocks: [TalkieBlock]
    private let eps: Float
    init(_ c: TalkieConfiguration, fused: Bool) {
        _embed.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        blocks = (0..<c.hiddenLayers).map { _ in TalkieBlock(c, fused: fused) }
        eps = c.rmsNormEpsilon
        super.init()
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let embedding = talkieNorm(embed(inputs), eps: eps)
        var h = embedding
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, block) in blocks.enumerated() { h = block(embedding, h, mask: mask, cache: cache?[index]) }
        return talkieNorm(h, eps: eps)
    }
}

public final class TalkieModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let usesFusedProjections: Bool
    private let configuration: TalkieConfiguration
    @ModuleInfo(key: "model") private var inner: TalkieInner
    @ModuleInfo(key: "lm_head") private var lmHead: Linear

    public init(_ configuration: TalkieConfiguration, fuseProjections: Bool = false) {
        self.configuration = configuration
        vocabularySize = configuration.vocabularySize
        kvHeads = Array(repeating: configuration.keyValueHeads, count: configuration.hiddenLayers)
        // Existing row-concatenation fusion assumes equal Q/K/V widths.
        // GQA derivatives retain separate source projections.
        usesFusedProjections =
            fuseProjections && configuration.permitsProjectionFusion
            && configuration.keyValueHeads == configuration.attentionHeads
        _inner.wrappedValue = TalkieInner(configuration, fused: usesFusedProjections)
        _lmHead.wrappedValue = Linear(configuration.hiddenSize, configuration.vocabularySize, bias: false)
        super.init()
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(inner(inputs, cache: cache))
    }
    public var loraLayers: [Module] { inner.blocks }
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try (0..<configuration.hiddenLayers).map { _ in
            try makeHybridAttentionKVCache(parameters: parameters, slidingWindow: nil, usesSlidingWindow: false)
        }
    }
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights.filter {
            !$0.key.contains("rotary") && !$0.key.hasSuffix(".cos") && !$0.key.hasSuffix(".sin")
        }
        if let bareHead = result.removeValue(forKey: "lm_head") {
            // Conflicting canonical and linear heads must fail strict loading.
            if result["lm_head.weight"] != nil {
                result["lm_head"] = bareHead
            } else {
                result["lm_head.weight"] = bareHead
            }
        }
        if let gain = result["lm_head_gain.w_g"], let head = result["lm_head.weight"], result["lm_head.scales"] == nil {
            // Authors apply the gain to weights before F.linear. Fold once, retaining
            // BF16 rounding; applying it to logits would change canonical BF16 outputs.
            result["lm_head.weight"] = head * gain.asType(head.dtype)
            result.removeValue(forKey: "lm_head_gain.w_g")
        }
        guard usesFusedProjections else { return result }
        for i in 0..<configuration.hiddenLayers {
            let prefix = "model.blocks.\(i)"
            Self.fuse(
                &result,
                sources: ["\(prefix).attn.attn_query", "\(prefix).attn.attn_key", "\(prefix).attn.attn_value"],
                destination: "\(prefix).attn.attn_qkv", rows: configuration.hiddenSize)
            Self.fuse(
                &result, sources: ["\(prefix).mlp.mlp_gate", "\(prefix).mlp.mlp_linear"],
                destination: "\(prefix).mlp.mlp_gate_up", rows: configuration.intermediateSize)
        }
        return result
    }
    private static func fuse(_ weights: inout [String: MLXArray], sources: [String], destination: String, rows: Int) {
        // Concatenate whole rows, including packed q4 words/scales/affine biases.
        // Missing or mixed representations remain for strict weight validation.
        let suffixes = ["weight", "scales", "biases"]
        let present = suffixes.filter { suffix in sources.contains { weights["\($0).\(suffix)"] != nil } }
        guard present.contains("weight"),
            present.allSatisfy({ suffix in
                weights["\(destination).\(suffix)"] == nil && sources.allSatisfy { weights["\($0).\(suffix)"] != nil }
            })
        else { return }
        for suffix in present {
            let arrays = sources.map { weights["\($0).\(suffix)"]! }
            guard
                arrays.allSatisfy({
                    $0.ndim == 2 && $0.dim(0) == rows && $0.dtype == arrays[0].dtype && $0.dim(1) == arrays[0].dim(1)
                })
            else { return }
        }
        for suffix in present {
            weights["\(destination).\(suffix)"] = concatenated(
                sources.map { weights.removeValue(forKey: "\($0).\(suffix)")! }, axis: 0)
        }
    }
}

public enum TalkieLoadingOptions {
    // Adapters retain the original projection names. Task-local configuration
    // avoids changing another model load when an adapter is loaded concurrently.
    @TaskLocal public static var fuseProjections = true

    public static func additionalEOSTokens(configuration: Data) -> Set<String> {
        struct Identity: Decodable {
            let modelType: String
            enum CodingKeys: String, CodingKey { case modelType = "model_type" }
        }
        guard (try? JSONDecoder().decode(Identity.self, from: configuration).modelType) == "talkie" else { return [] }
        return ["<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]
    }
}

public enum TalkieModelRegistration {
    public static func register() async {
        await LLMTypeRegistry.shared.registerModelType(
            "talkie",
            creator: { data in
                TalkieModel(
                    try JSONDecoder().decode(TalkieConfiguration.self, from: data),
                    // Keep fusion opt-in until representative paired measurements establish
                    // a reliable benefit. The initial Q8 runs preserved output but drifted.
                    fuseProjections: TalkieLoadingOptions.fuseProjections
                        && ProcessInfo.processInfo.environment["MIDNIGHT_TALKIE_FUSE_PROJECTIONS"] == "1")
            })
    }
    public static func isRegistered() async -> Bool { await LLMTypeRegistry.shared.contains("talkie") }
}

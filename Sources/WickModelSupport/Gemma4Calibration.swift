import Foundation
import MLX
@_spi(GemmaEncoder) import MLXLLM
import MLXLMCommon
import MLXNN

public enum Gemma4CalibrationError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidWeights(String)
    case invalidInput(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail): "Invalid Gemma calibration configuration: \(detail)"
        case .invalidWeights(let detail): "Invalid Gemma calibration weights: \(detail)"
        case .invalidInput(let detail): "Invalid Gemma calibration input: \(detail)"
        }
    }
}

/// The native configuration and geometry needed to run one BF16 teacher layer.
/// Supports the no-PLE, no-KV-sharing geometry of Gemma 4 A4B and 31B. No whole
/// teacher is constructed or materialized by this descriptor.
public struct Gemma4CalibrationConfiguration {
    public let hiddenSize: Int
    public let layerCount: Int
    public let layerTypes: [String]
    public let slidingWindow: Int
    public let moduleRoot: String
    fileprivate let native: Gemma4TextConfiguration

    public init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modelType = root["model_type"] as? String,
            ["gemma4", "gemma4_text"].contains(modelType)
        else { throw Gemma4CalibrationError.invalidConfiguration("expected gemma4 or gemma4_text") }
        let text = root["text_config"] as? [String: Any] ?? root
        for candidate in [root, text] {
            for name in ["quantization", "quantization_config"] {
                if let value = candidate[name], !(value is NSNull) {
                    throw Gemma4CalibrationError.invalidConfiguration("quantized teachers are unsupported")
                }
            }
        }
        guard let width = text["hidden_size"] as? Int, width > 0,
            let count = text["num_hidden_layers"] as? Int, count > 0,
            let types = text["layer_types"] as? [String], types.count == count,
            types.allSatisfy({ ["sliding_attention", "full_attention"].contains($0) }),
            let window = text["sliding_window"] as? Int, window > 0,
            (text["num_kv_shared_layers"] as? Int) == 0,
            (text["hidden_size_per_layer_input"] as? Int) == 0
        else {
            throw Gemma4CalibrationError.invalidConfiguration(
                "require positive geometry, declared layer types, and no PLE or shared KV layers")
        }
        let epsilon = Float((text["rms_norm_eps"] as? Double) ?? 0.000001)
        let positiveDimensions = [
            "intermediate_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "global_head_dim",
        ]
        guard epsilon == Float(0.000001),
            positiveDimensions.allSatisfy({
                guard let value = text[$0] as? Int else { return false }
                return value > 0
            })
        else {
            throw Gemma4CalibrationError.invalidConfiguration(
                "require positive explicit projection dimensions and native RMS epsilon 1e-6")
        }
        if text["enable_moe_block"] as? Bool == true {
            guard let experts = text["num_experts"] as? Int, experts > 0,
                let topK = text["top_k_experts"] as? Int, topK > 0, topK <= experts,
                let intermediate = text["moe_intermediate_size"] as? Int, intermediate > 0
            else { throw Gemma4CalibrationError.invalidConfiguration("invalid routed expert geometry") }
        }
        var nativeText = text
        if let vocabulary = root["vocab_size"] { nativeText["vocab_size"] = vocabulary }
        native = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: JSONSerialization.data(withJSONObject: nativeText))
        hiddenSize = width
        layerCount = count
        layerTypes = types
        slidingWindow = window
        moduleRoot = modelType == "gemma4" ? "language_model.model" : "model"
    }
}

/// Runs the unchanged native decoder on one selectively loaded BF16 layer.
/// The caller owns layer-major activation spooling and releases each block before
/// loading the next. Observers run synchronously on the caller's serialized MLX
/// worker; no process-wide hook or serving model is changed.
public final class Gemma4CalibrationBlock {
    public let module: Module
    public let modulePrefix: String
    public let denseProjectionWidths: [String: Int]
    public let routedProjectionPaths: Set<String>
    private let layer: Gemma4DecoderLayer
    private let configuration: Gemma4CalibrationConfiguration
    private let layerIndex: Int

    public init(
        configuration: Gemma4CalibrationConfiguration,
        layerIndex: Int,
        sourceWeights: [String: MLXArray],
        observeDense: ((String, MLXArray) -> Void)? = nil,
        observeRouted: LagunaRoutedActivationObserver? = nil
    ) throws {
        guard (0..<configuration.layerCount).contains(layerIndex) else {
            throw Gemma4CalibrationError.invalidConfiguration("layer index out of range")
        }
        self.configuration = configuration
        self.layerIndex = layerIndex
        modulePrefix = configuration.moduleRoot + ".layers.\(layerIndex)"
        let weights = try Self.normalize(sourceWeights, layerIndex: layerIndex)
        layer = Gemma4DecoderLayer(configuration.native, layerIdx: layerIndex)
        try layer.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        layer.train(false)
        module = layer
        var widths = [String: Int]()
        var routed = Set<String>()
        var replacements = [(String, Module)]()
        for (path, child) in layer.leafModules().flattened() {
            let absolutePath = modulePrefix + "." + path
            if let linear = child as? Linear {
                guard !(linear is Quantized), linear.weight.dtype == .bfloat16,
                    linear.weight.ndim == 2, linear.weight.dim(-1) > 0
                else { throw Gemma4CalibrationError.invalidWeights("\(absolutePath) requires BF16 dense weights") }
                widths[absolutePath] = linear.weight.dim(-1)
                if let observeDense {
                    replacements.append(
                        (
                            path,
                            GemmaDenseActivationTap(
                                linear: linear, path: absolutePath, observe: observeDense)
                        ))
                }
            } else if let linear = child as? SwitchLinear {
                let parameters = Dictionary(uniqueKeysWithValues: linear.parameters().flattened())
                guard !(linear is Quantized), let weight = parameters["weight"],
                    weight.dtype == .bfloat16, weight.ndim == 3,
                    weight.shape.allSatisfy({ $0 > 0 })
                else { throw Gemma4CalibrationError.invalidWeights("\(absolutePath) requires BF16 expert weights") }
                routed.insert(absolutePath)
                if let observeRouted {
                    replacements.append(
                        (
                            path,
                            GemmaRoutedActivationTap(
                                weight: weight, bias: parameters["bias"], path: absolutePath, observer: observeRouted)
                        ))
                }
            }
        }
        guard !widths.isEmpty else { throw Gemma4CalibrationError.invalidWeights("no dense projections") }
        denseProjectionWidths = widths
        routedProjectionPaths = routed
        if !replacements.isEmpty {
            try layer.update(modules: ModuleChildren.unflattened(replacements), verify: [.noUnusedKeys])
        }
        layer.train(false)
    }

    public func callAsFunction(_ hidden: MLXArray) throws -> MLXArray {
        guard hidden.ndim == 3, hidden.dim(0) > 0, hidden.dim(1) > 0,
            hidden.dim(-1) == configuration.hiddenSize, hidden.dtype == .bfloat16
        else { throw Gemma4CalibrationError.invalidInput("expected nonempty BF16 [batch, tokens, hidden] input") }
        if !routedProjectionPaths.isEmpty, Device.defaultDevice().deviceType == .cpu {
            throw Gemma4CalibrationError.invalidInput(
                "BF16 expert gather is unsupported on the CPU backend; use Metal for A4B calibration")
        }
        let window =
            configuration.layerTypes[layerIndex] == "sliding_attention"
            ? configuration.slidingWindow : nil
        let mask = createAttentionMask(h: hidden, cache: nil, windowSize: window)
        return layer(hidden, mask: mask).0
    }

    private static func normalize(
        _ source: [String: MLXArray], layerIndex: Int
    ) throws -> [String: MLXArray] {
        let prefixes = [
            "model.language_model.layers.", "language_model.model.layers.",
            "language_model.layers.", "model.layers.",
        ].map { $0 + "\(layerIndex)." }
        var result = [String: MLXArray]()
        func insert(_ key: String, _ value: MLXArray) throws {
            guard result[key] == nil else {
                throw Gemma4CalibrationError.invalidWeights("duplicate source alias for \(key)")
            }
            result[key] = value
        }
        for (key, value) in source.sorted(by: { $0.key < $1.key }) {
            guard let prefix = prefixes.first(where: { key.hasPrefix($0) }) else {
                throw Gemma4CalibrationError.invalidWeights("tensor outside selected layer: \(key)")
            }
            let relative = String(key.dropFirst(prefix.count))
            if relative.contains("self_attn.rotary_emb")
                || ["input_max", "input_min", "output_max", "output_min"].contains(where: { relative.contains($0) })
            {
                continue
            }
            if relative == "experts.gate_up_proj" {
                guard value.ndim == 3, value.dim(-2) > 0, value.dim(-2) % 2 == 0 else {
                    throw Gemma4CalibrationError.invalidWeights("invalid fused expert gate/up shape")
                }
                let midpoint = value.dim(-2) / 2
                try insert("experts.switch_glu.gate_proj.weight", value[.ellipsis, ..<midpoint, 0...])
                try insert("experts.switch_glu.up_proj.weight", value[.ellipsis, midpoint..., 0...])
            } else if relative == "experts.down_proj" {
                try insert("experts.switch_glu.down_proj.weight", value)
            } else {
                try insert(relative, value)
            }
        }
        return result
    }
}

private final class GemmaDenseActivationTap: Linear {
    private let path: String
    private let observe: (String, MLXArray) -> Void

    init(linear: Linear, path: String, observe: @escaping (String, MLXArray) -> Void) {
        self.path = path
        self.observe = observe
        super.init(weight: linear.weight, bias: linear.bias)
        train(linear.training)
    }

    override func callAsFunction(_ input: MLXArray) -> MLXArray {
        observe(path, input)
        return super.callAsFunction(input)
    }
}

private final class GemmaRoutedActivationTap: SwitchLinear {
    private let path: String
    private let observer: LagunaRoutedActivationObserver
    private let expertCount: Int

    init(weight: MLXArray, bias: MLXArray?, path: String, observer: LagunaRoutedActivationObserver) {
        self.path = path
        self.observer = observer
        expertCount = weight.dim(0)
        super.init(
            inputDims: weight.dim(2), outputDims: weight.dim(1), numExperts: weight.dim(0),
            weight: weight, bias: bias)
    }

    override func callAsFunction(
        _ input: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        observer.observeRoutedProjection(path: path, input: input, indices: indices, expertCount: expertCount)
        return super.callAsFunction(input, indices, sortedIndices: sortedIndices)
    }
}

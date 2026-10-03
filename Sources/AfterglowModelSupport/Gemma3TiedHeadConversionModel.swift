import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// A conversion-only tree that retains Gemma 3's shared embedding once on disk.
///
/// Native runtime loading recreates its head alias from the embedding triplet.
public final class Gemma3TiedHeadConversionModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "model") var model: Gemma3Model
    private let sanitizeSource: ([String: MLXArray]) -> [String: MLXArray]

    public init(_ source: Gemma3TextModel) {
        _model.wrappedValue = source.model
        // Store a closure rather than the native model as another Module child.
        sanitizeSource = { source.sanitize(weights: $0) }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitizeSource(weights).filter { !$0.key.split(separator: ".").contains("lm_head") }
    }
}

import DecisionModels
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

@testable import DecisionMLX

final class DecisionGradientTests: XCTestCase {
    func testQwenHybridLayersHaveFiniteCandidateGradientsAndFrozenBase() throws {
        try Device.withDefaultDevice(.cpu) {
            try GatedDeltaExecution.$useMetalKernel.withValue(false) {
                let configuration = try JSONDecoder().decode(
                    Qwen35TextConfiguration.self,
                    from: Data(
                        #"{"model_type":"qwen3_5_text","hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"linear_num_value_heads":2,"linear_num_key_heads":1,"linear_key_head_dim":32,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,"vocab_size":64,"full_attention_interval":2,"tie_word_embeddings":true}"#
                            .utf8))
                MLXRandom.seed(17)
                let model = Qwen35TextModel(configuration)
                let lora = LoRAConfiguration(numLayers: 2, loraParameters: .init(rank: 2, scale: 2, dropout: 0))
                _ = try LoRAContainer.from(model: model, configuration: lora)
                let before = Dictionary(
                    uniqueKeysWithValues: model.parameters().flattened().filter { !$0.0.contains(".lora_") }.map {
                        ($0.0, $0.1 + 0)
                    })
                eval(Array(before.values))
                let module: Module = model
                let gradient = valueAndGrad(model: module) { model, _ in
                    let logits = (model as! Qwen35TextModel)(MLXArray([1, 2, 3, 4]).reshaped(1, 4), cache: nil)[
                        0, -1, 0...]
                    let candidates = take(logits, MLXArray([32, 33]), axis: 0).reshaped(1, 2)
                    return [crossEntropy(logits: candidates, targets: MLXArray([1])).mean()]
                }
                let (loss, gradients) = gradient(module, [])
                eval(loss, gradients)
                XCTAssertTrue(loss[0].item(Float.self).isFinite)
                let tensors = gradients.flattened()
                XCTAssertFalse(tensors.isEmpty)
                XCTAssertTrue(tensors.allSatisfy { $0.0.contains(".lora_") })
                let recurrent = tensors.filter { $0.0.contains("linear_attn") }
                XCTAssertFalse(recurrent.isEmpty)
                let norm = recurrent.reduce(Float(0)) { $0 + square($1.1).sum().item(Float.self) }
                XCTAssertTrue(norm.isFinite && norm > 0)
                let parameters = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
                let updated = Dictionary(uniqueKeysWithValues: tensors.map { ($0.0, parameters[$0.0]! - 0.01 * $0.1) })
                try model.update(parameters: .unflattened(updated), verify: [.noUnusedKeys, .shapeMismatch])
                eval(model)
                for (key, value) in model.parameters().flattened() where before[key] != nil {
                    XCTAssertEqual(abs(value - before[key]!).max().item(Float.self), 0)
                }
            }
        }
    }
}

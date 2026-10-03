import Foundation
import MLXLMCommon
import Testing

@testable import LagunaScaleSearchCore

struct LagunaGroupSizePolicyTests {
  @Test("Native fused Laguna modules inherit G128 while routers and tied embedding retain G64")
  func nativeFusedPolicyResolution() throws {
    let prefix = "language_model.model"
    let router = "\(prefix).layers.1.mlp.gate.proj"
    let embedding = "\(prefix).embed_tokens"
    let splitModules = [
      "\(prefix).layers.0.mlp.gate_proj", "\(prefix).layers.0.mlp.up_proj",
      "\(prefix).layers.1.mlp.switch_mlp.gate_proj", "\(prefix).layers.1.mlp.switch_mlp.up_proj",
      "\(prefix).layers.1.mlp.shared_expert.gate_proj", "\(prefix).layers.1.mlp.shared_expert.up_proj",
    ]
    let template = try JSONSerialization.data(withJSONObject: [
      "model_type": "laguna", "quantization": [
        "group_size": 64, "bits": 4, router: ["group_size": 64, "bits": 8]]])
    let policy = try LagunaGroupSizePolicy(templateConfiguration: template,
      groupSize: 128, q4Modules: splitModules, q8Modules: [router], embeddings: [embedding])
    let config = try JSONDecoder().decode(BaseConfiguration.self, from: policy.configuration)
    let quantization = try #require(config.perLayerQuantization)
    // Laguna sanitation concatenates split pairs before loadWeights resolves native paths.
    for path in ["\(prefix).layers.0.mlp.gate_up_proj",
      "\(prefix).layers.1.mlp.switch_mlp.gate_up_proj",
      "\(prefix).layers.1.mlp.shared_expert.gate_up_proj"]
    {
      let geometry = try #require(quantization.quantization(layer: path))
      #expect(geometry.groupSize == 128)
      #expect(geometry.bits == 4)
    }
    #expect(quantization.quantization(layer: router)?.groupSize == 64)
    #expect(quantization.quantization(layer: router)?.bits == 8)
    #expect(quantization.quantization(layer: embedding)?.groupSize == 64)
    #expect(quantization.quantization(layer: embedding)?.bits == 4)
  }

  @Test("G128 conversion refuses a G128 template or a changed preserved-router policy")
  func rejectsIncompatibleTemplatePolicy() throws {
    let dense = "language_model.model.layers.0.self_attn.q_proj"
    let router = "language_model.model.layers.1.mlp.gate.proj"
    for (globalGroup, routerBits) in [(128, 8), (64, 4)] {
      let quantization: [String: Any] = ["group_size": globalGroup, "bits": 4,
        router: ["group_size": 64, "bits": routerBits]]
      let data = try JSONSerialization.data(withJSONObject: ["quantization": quantization])
      #expect(throws: LagunaActivationInputError.self) {
        try LagunaGroupSizePolicy(templateConfiguration: data, groupSize: 128,
          q4Modules: [dense], q8Modules: [router], embeddings: [])
      }
    }
  }

  @Test("Changed tensor totals preserve unrelated index metadata and shard placement")
  func indexAccounting() throws {
    let input = try JSONSerialization.data(withJSONObject: [
      "metadata": ["total_size": 4096, "source": "fixture"],
      "weight_map": ["a.weight": "shard-1.safetensors", "a.scales": "shard-2.safetensors"]])
    let output = try LagunaGroupSizePolicy.updatedIndex(input, tensorBytes: 3072)
    let decoded = try #require(JSONSerialization.jsonObject(with: output) as? [String: Any])
    let metadata = try #require(decoded["metadata"] as? [String: Any])
    #expect(metadata["total_size"] as? Int == 3072)
    #expect(metadata["source"] as? String == "fixture")
    #expect(decoded["weight_map"] as? [String: String]
      == ["a.weight": "shard-1.safetensors", "a.scales": "shard-2.safetensors"])
  }
}

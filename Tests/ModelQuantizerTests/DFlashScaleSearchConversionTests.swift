import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import WickModelSupport

final class DFlashScaleSearchConversionTests: XCTestCase {
  func testDFlashFusedCheckpointConvertsAndReloads() async throws {
    try await Device.withDefaultDevice(.cpu) {
      let fileManager = FileManager.default
      let root = fileManager.temporaryDirectory.appendingPathComponent(
        "model-runner-dflash-conversion-\(UUID().uuidString)")
      let source = root.appendingPathComponent("source")
      let output = root.appendingPathComponent("output")
      try fileManager.createDirectory(at: source, withIntermediateDirectories: true)
      defer { try? fileManager.removeItem(at: root) }

      let config = """
        {
          "architectures": ["DFlashLagunaForCausalLM"],
          "attention_bias": false,
          "head_dim": 32,
          "hidden_size": 64,
          "intermediate_size": 128,
          "max_position_embeddings": 128,
          "model_type": "laguna",
          "num_attention_heads": 2,
          "num_hidden_layers": 1,
          "num_key_value_heads": 1,
          "rms_norm_eps": 0.000001,
          "rope_theta": 500000,
          "sliding_window": 8,
          "vocab_size": 128,
          "layer_types": ["sliding_attention"],
          "dflash_config": {
            "block_size": 4,
            "mask_token_id": 3,
            "num_target_layers": 4,
            "target_layer_ids": [0, 2],
            "causal": true
          }
        }
        """
      let configData = Data(config.utf8)
      try configData.write(to: source.appendingPathComponent("config.json"))

      let configuration = try JSONDecoder().decode(
        LagunaDFlashConfiguration.self, from: configData)
      let sourceModel = LagunaDFlashModel(configuration)
      var sourceArrays = Dictionary(
        uniqueKeysWithValues: sourceModel.parameters().flattened())

      let attention = "layers.0.self_attn"
      let query = try XCTUnwrap(sourceArrays.removeValue(forKey: "\(attention).q_proj.weight"))
      let keyValue = try XCTUnwrap(
        sourceArrays.removeValue(forKey: "\(attention).kv_proj.weight"))
      sourceArrays["\(attention).qkv_proj.weight"] = concatenated(
        [query, keyValue], axis: 0)

      let mlp = "layers.0.mlp"
      let gateUp = try XCTUnwrap(
        sourceArrays.removeValue(forKey: "\(mlp).gate_up_proj.weight"))
      let gateUpParts = MLX.split(gateUp, parts: 2, axis: 0)
      sourceArrays["\(mlp).gate_proj.weight"] = gateUpParts[0]
      sourceArrays["\(mlp).up_proj.weight"] = gateUpParts[1]

      try save(
        arrays: sourceArrays,
        metadata: ["format": "pt"],
        url: source.appendingPathComponent("model.safetensors"),
        stream: .cpu
      )

      let q8Paths: Set<String> = ["fc", "layers.0.self_attn.g_proj"]
      let q8 = ModelConversionQuantization(
        bits: 8,
        groupSize: 64,
        mode: .affine,
        calibration: .standard
      )
      let result = try MLXLMCommon.convert(
        modelDirectory: source,
        model: LagunaDFlashModel(configuration),
        to: output,
        options: ModelConversionOptions(
          bits: 4,
          groupSize: 64,
          mode: .affine,
          calibration: .q4AffineScaleSearch,
          maxShardSize: 16 * 1_024 * 1_024,
          quantizationPredicate: { path, _ in
            q8Paths.contains(path) ? .quantize(q8) : .quantize()
          }
        )
      )

      XCTAssertFalse(result.weightsURLs.isEmpty)
      let outputConfigData = try Data(
        contentsOf: output.appendingPathComponent("config.json"))
      let outputConfig = try JSONDecoder.json5().decode(
        BaseConfiguration.self, from: outputConfigData)
      let quantization = try XCTUnwrap(outputConfig.perLayerQuantization)
      XCTAssertEqual(quantization.quantization?.bits, 4)
      XCTAssertEqual(quantization.quantization?.groupSize, 64)
      for path in q8Paths {
        guard case .quantize(let override)? = quantization.perLayerQuantization[path] else {
          return XCTFail("missing DFlash Q8 override for \(path)")
        }
        XCTAssertEqual(override.bits, 8)
        XCTAssertEqual(override.groupSize, 64)
      }

      var outputArrays = [String: MLXArray]()
      for weightsURL in result.weightsURLs {
        outputArrays.merge(
          try loadArrays(url: weightsURL, stream: .cpu),
          uniquingKeysWith: { _, replacement in replacement }
        )
      }
      XCTAssertNil(outputArrays["\(attention).qkv_proj.weight"])
      XCTAssertNotNil(outputArrays["\(attention).q_proj.scales"])
      XCTAssertNotNil(outputArrays["\(attention).kv_proj.biases"])
      XCTAssertNil(outputArrays["\(mlp).gate_proj.weight"])
      XCTAssertNil(outputArrays["\(mlp).up_proj.weight"])
      XCTAssertNotNil(outputArrays["\(mlp).gate_up_proj.scales"])
      XCTAssertNotNil(outputArrays["fc.scales"])

      await LagunaDFlashRegistration.register()
      let reloaded = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: outputConfigData,
        modelType: "laguna"
      )
      try loadWeights(
        modelDirectory: output,
        model: reloaded,
        perLayerQuantization: outputConfig.perLayerQuantization
      )
    }
  }
}

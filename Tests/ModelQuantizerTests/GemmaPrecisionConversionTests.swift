import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class GemmaPrecisionConversionTests: XCTestCase {
    func testGemma3PrecisionCandidatesPreserveTiedHeadAndReload() throws {
        try verifyCandidates(modelType: "gemma3_text", moe: false)
    }

    func testGemma4DensePrecisionCandidatesReloadFromCLI() throws {
        try verifyCandidates(modelType: "gemma4_text", moe: false)
    }

    func testGemma4MoEPrecisionCandidatesPreserveQ8RouterAndReloadFromCLI() throws {
        try verifyCandidates(modelType: "gemma4_text", moe: true)
    }

    private func verifyCandidates(modelType: String, moe: Bool) throws {
        let executable =
            ProcessInfo.processInfo.environment["WICK_TEST_EXECUTABLE"]
            ?? FileManager.default.currentDirectoryPath + "/.build/release/wick"
        guard FileManager.default.isExecutableFile(atPath: executable) else {
            throw XCTSkip("Set WICK_TEST_EXECUTABLE to the built Wick CLI")
        }
        try Device.withDefaultDevice(.cpu) {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            let source = root.appendingPathComponent("source")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let configuration: [String: Any] = [
                "model_type": modelType, "vocab_size": 32, "hidden_size": 128,
                "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "num_global_key_value_heads": 2,
                "head_dim": 32, "global_head_dim": 32, "sliding_window": 128,
                "layer_types": ["sliding_attention", "full_attention"],
                "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
                "attention_k_eq_v": false, "use_double_wide_mlp": false,
                "enable_moe_block": moe, "num_experts": 4, "top_k_experts": 2,
                "moe_intermediate_size": 64, "tie_word_embeddings": true,
            ]
            let config = try JSONSerialization.data(withJSONObject: configuration, options: [.sortedKeys])
            try config.write(to: source.appendingPathComponent("config.json"))
            let native = try makeModel(modelType: modelType, configuration: config)
            var arrays = Dictionary(
                uniqueKeysWithValues: native.parameters().flattened().map { name, tensor in
                    let values = (0..<tensor.size).map { Float($0 % 37 - 18) / 41 }
                    return (name, MLXArray(values).reshaped(tensor.shape).asType(.bfloat16))
                })
            if modelType == "gemma3_text" {
                arrays["lm_head.weight"] = arrays["model.embed_tokens.weight"]
            }
            let sourceWeights = source.appendingPathComponent("model.safetensors")
            try save(arrays: arrays, url: sourceWeights)
            let sourceBytes = try Data(contentsOf: sourceWeights)
            let recipes = [
                (4, 32, false, "standard"), (5, 64, false, "standard"), (5, 32, false, "standard"),
                (4, 64, true, "standard"), (5, 64, false, "scale-search"),
            ]
            for (bits, groupSize, selective, calibration) in recipes {
                let output = root.appendingPathComponent("q\(bits)-g\(groupSize)-\(selective)-\(calibration)")
                var arguments = [
                    source.path, output.path, "--gemma-group-policy", "--mode", "affine",
                    "--bits", String(bits), "--group-size", String(groupSize),
                    "--calibration", calibration, "--bounded-memory", "--cpu",
                ]
                if selective { arguments += ["--g128-module", "*.mlp.down_proj"] }
                if moe { arguments += ["--q8-module", "*.router.proj"] }
                try runCLI(executable: executable, arguments: arguments)
                let exportedConfig = try Data(contentsOf: output.appendingPathComponent("config.json"))
                var exported = [String: MLXArray]()
                for file in try FileManager.default.contentsOfDirectory(at: output, includingPropertiesForKeys: nil)
                where file.pathExtension == "safetensors" {
                    exported.merge(try loadArrays(url: file)) { _, new in new }
                }
                let query = "model.layers.0.self_attn.q_proj"
                let packed = try XCTUnwrap(exported[query + ".weight"])
                let original = try XCTUnwrap(arrays[query + ".weight"])
                XCTAssertEqual(packed.shape.last, original.shape.last! / 32 * bits)
                let scales = try XCTUnwrap(exported[query + ".scales"])
                XCTAssertEqual(scales.shape.last, original.shape.last! / groupSize)
                XCTAssertEqual(scales.dtype, .bfloat16)
                for (name, tensor) in arrays where name.contains("norm") {
                    let actual = try XCTUnwrap(exported[name])
                    XCTAssertTrue(MLX.all(actual .== tensor).item(Bool.self), name)
                }
                if selective {
                    let path = "model.layers.0.mlp.down_proj"
                    let expected = MLX.quantized(try XCTUnwrap(arrays[path + ".weight"]), groupSize: 128, bits: 4)
                    XCTAssertTrue(MLX.all(try XCTUnwrap(exported[path + ".weight"]) .== expected.wq).item(Bool.self))
                    XCTAssertTrue(
                        MLX.all(try XCTUnwrap(exported[path + ".scales"]) .== expected.scales).item(Bool.self))
                }
                if calibration == "scale-search" {
                    let provenanceData = try Data(contentsOf: output.appendingPathComponent("quantization.json"))
                    let provenance = try XCTUnwrap(
                        try JSONSerialization.jsonObject(with: provenanceData) as? [String: Any])
                    XCTAssertEqual(provenance["algorithm"] as? String, "q5r8_affine_scale_search_ls2")
                    XCTAssertEqual(provenance["q4_scale_search_modules"] as? [String], [])
                    let searched = try XCTUnwrap(provenance["q5_scale_search_modules"] as? [String])
                    XCTAssertTrue(searched.contains(query))
                    XCTAssertFalse(searched.contains { $0.contains("embed_tokens") || $0.contains("router") })
                }
                let restored = try makeModel(modelType: modelType, configuration: exportedConfig)
                let base = try JSONDecoder().decode(BaseConfiguration.self, from: exportedConfig)
                try loadWeights(
                    modelDirectory: output, model: restored, perLayerQuantization: base.perLayerQuantization)
                if modelType == "gemma3_text" {
                    XCTAssertFalse(exported.keys.contains { $0.hasPrefix("lm_head.") })
                    let leaves = Dictionary(uniqueKeysWithValues: restored.leafModules().flattened())
                    let head = try XCTUnwrap(leaves["lm_head"] as? QuantizedLinear)
                    let embedding = try XCTUnwrap(leaves["model.embed_tokens"] as? QuantizedEmbedding)
                    XCTAssertEqual(head.bits, bits)
                    XCTAssertEqual(head.groupSize, groupSize)
                    let hidden = MLX.ones([1, 128], dtype: .bfloat16)
                    XCTAssertTrue(MLX.all(head(hidden) .== embedding.asLinear(hidden)).item(Bool.self))
                }
                if moe {
                    let leaves = Dictionary(uniqueKeysWithValues: restored.leafModules().flattened())
                    let router = try XCTUnwrap(leaves["model.layers.0.router.proj"] as? QuantizedLinear)
                    XCTAssertEqual(router.bits, 8)
                    XCTAssertEqual(router.groupSize, 64)
                    XCTAssertTrue(exported.keys.contains { $0.contains("experts") && $0.hasSuffix(".weight") })
                }
                let logits = restored(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: nil)
                XCTAssertEqual(logits.shape, [1, 2, 32])
                XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
                XCTAssertEqual(try Data(contentsOf: sourceWeights), sourceBytes)
            }
        }
    }

    private func makeModel(modelType: String, configuration: Data) throws -> any LLMModel {
        if modelType == "gemma3_text" {
            return Gemma3TextModel(try JSONDecoder().decode(Gemma3TextConfiguration.self, from: configuration))
        }
        return Gemma4TextModel(try JSONDecoder().decode(Gemma4TextConfiguration.self, from: configuration))
    }

    private func runCLI(executable: String, arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertEqual(process.terminationReason, .exit, text)
        XCTAssertEqual(process.terminationStatus, 0, text)
        guard process.terminationReason == .exit, process.terminationStatus == 0 else {
            throw NSError(
                domain: "GemmaPrecisionConversionTests", code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: text])
        }
    }
}

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import WickModelSupport

final class DFlashFormatTests: XCTestCase {
    func testMXFP4CLIConversionProtectsQ8ModulesAndDraftsAfterReload() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let executable = environment["WICK_TEST_EXECUTABLE"] ?? environment["FACET_TEST_EXECUTABLE"],
            !executable.isEmpty
        else {
            throw XCTSkip("Set WICK_TEST_EXECUTABLE to the built wick executable for CLI format tests.")
        }
        try await Device.withDefaultDevice(.cpu) {
            let manager = FileManager.default
            let root = manager.temporaryDirectory.appendingPathComponent("wick-dflash-mxfp4-\(UUID())")
            let source = root.appendingPathComponent("source")
            let output = root.appendingPathComponent("output")
            try manager.createDirectory(at: source, withIntermediateDirectories: true)
            defer { try? manager.removeItem(at: root) }

            let fixtureURL = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Fixtures/LagunaDFlashTiny/config.json")
            let sourceConfig = try Data(contentsOf: fixtureURL)
            try sourceConfig.write(to: source.appendingPathComponent("config.json"))
            let configuration = try JSONDecoder().decode(LagunaDFlashConfiguration.self, from: sourceConfig)
            let sourceModel = LagunaDFlashModel(configuration)
            let originals = Dictionary(uniqueKeysWithValues: sourceModel.parameters().flattened())
            var sourceWeights = originals
            let attention = "layers.0.self_attn"
            let query = try XCTUnwrap(sourceWeights.removeValue(forKey: attention + ".q_proj.weight"))
            let keyValue = try XCTUnwrap(sourceWeights.removeValue(forKey: attention + ".kv_proj.weight"))
            sourceWeights[attention + ".qkv_proj.weight"] = concatenated([query, keyValue], axis: 0)
            let mlp = "layers.0.mlp"
            let gateUp = try XCTUnwrap(sourceWeights.removeValue(forKey: mlp + ".gate_up_proj.weight"))
            let parts = MLX.split(gateUp, parts: 2, axis: 0)
            sourceWeights[mlp + ".gate_proj.weight"] = parts[0]
            sourceWeights[mlp + ".up_proj.weight"] = parts[1]
            try save(
                arrays: sourceWeights, metadata: ["format": "pt"],
                url: source.appendingPathComponent("model.safetensors"), stream: .cpu
            )

            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = ["quantize", source.path, output.path, "--mode", "mxfp4", "--cpu"]
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            try process.run()
            let outputData = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let log = String(decoding: outputData, as: UTF8.self)
            XCTAssertEqual(process.terminationReason, .exit, log)
            XCTAssertEqual(process.terminationStatus, 0, log)
            guard process.terminationReason == .exit, process.terminationStatus == 0 else { return }

            let configData = try Data(contentsOf: output.appendingPathComponent("config.json"))
            let json = try XCTUnwrap(JSONSerialization.jsonObject(with: configData) as? [String: Any])
            let metadata = try XCTUnwrap(json["quantization"] as? [String: Any])
            XCTAssertEqual(metadata["mode"] as? String, "mxfp4")
            XCTAssertEqual(metadata["bits"] as? Int, 4)
            XCTAssertEqual(metadata["group_size"] as? Int, 32)
            let protectedPaths = ["fc", attention + ".g_proj"]
            for path in protectedPaths {
                let override = try XCTUnwrap(metadata[path] as? [String: Any])
                XCTAssertEqual(override["mode"] as? String, "affine", path)
                XCTAssertEqual(override["bits"] as? Int, 8, path)
                XCTAssertEqual(override["group_size"] as? Int, 64, path)
            }

            let shards = try manager.contentsOfDirectory(at: output, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "safetensors" }
            XCTAssertFalse(shards.isEmpty)
            var weights = [String: MLXArray]()
            for shard in shards {
                weights.merge(try loadArrays(url: shard, stream: .cpu), uniquingKeysWith: { _, new in new })
            }
            XCTAssertNil(weights[attention + ".qkv_proj.weight"])
            XCTAssertNil(weights[mlp + ".gate_proj.weight"])
            XCTAssertNil(weights[mlp + ".up_proj.weight"])
            for path in [attention + ".q_proj", attention + ".kv_proj", mlp + ".gate_up_proj"] {
                let original = try XCTUnwrap(originals[path + ".weight"])
                let expected = MLX.quantized(original, groupSize: 32, bits: 4, mode: .mxfp4)
                let packed = try XCTUnwrap(weights[path + ".weight"])
                let scales = try XCTUnwrap(weights[path + ".scales"])
                XCTAssertEqual(packed.dtype, .uint32, path)
                XCTAssertTrue(MLX.arrayEqual(packed, expected.wq).item(Bool.self), path)
                XCTAssertTrue(MLX.arrayEqual(scales, expected.scales).item(Bool.self), path)
                XCTAssertNil(weights[path + ".biases"], path)
            }
            for path in protectedPaths {
                let original = try XCTUnwrap(originals[path + ".weight"])
                let expected = MLX.quantized(original, groupSize: 64, bits: 8, mode: .affine)
                let packed = try XCTUnwrap(weights[path + ".weight"])
                let scales = try XCTUnwrap(weights[path + ".scales"])
                let biases = try XCTUnwrap(weights[path + ".biases"])
                XCTAssertTrue(MLX.arrayEqual(packed, expected.wq).item(Bool.self), path)
                XCTAssertTrue(MLX.arrayEqual(scales, expected.scales).item(Bool.self), path)
                XCTAssertTrue(MLX.arrayEqual(biases, try XCTUnwrap(expected.biases)).item(Bool.self), path)
            }

            let provenanceData = try Data(contentsOf: output.appendingPathComponent("quantization.json"))
            let provenance = try XCTUnwrap(JSONSerialization.jsonObject(with: provenanceData) as? [String: Any])
            XCTAssertEqual(provenance["mode"] as? String, "mxfp4")
            XCTAssertEqual(provenance["bits"] as? Int, 4)
            XCTAssertEqual(provenance["group_size"] as? Int, 32)
            XCTAssertEqual(provenance["q8_modules"] as? [String], protectedPaths)
            XCTAssertEqual(provenance["mandatory_q8_modules"] as? [String], protectedPaths)
            XCTAssertEqual(provenance["q4_scale_search_modules"] as? [String], [])
            XCTAssertTrue(
                try XCTUnwrap(provenance["standard_modules"] as? [String]).contains(mlp + ".gate_up_proj")
            )

            await LagunaDFlashRegistration.register()
            let reloaded = try await MTPDrafterTypeRegistry.shared.createModel(
                configuration: configData, modelType: "laguna"
            )
            let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
            try loadWeights(modelDirectory: output, model: reloaded, perLayerQuantization: base.perLayerQuantization)

            // DFlash shares its target's embedding and output head. This tiny
            // target provides those layers while the reloaded drafter computes
            // context projection, cached attention, MLP, and proposal logits.
            let target = try makeTarget()
            let sampler = CapturingSampler()
            let hidden = MLXArray((0..<128).map { Float($0 % 17 - 8) / 100 }).reshaped(1, 1, 128)
            let tokens = reloaded.draftBlock(
                target: target, lastToken: MLXArray([Int32(1)]), lastHidden: hidden,
                sharedKV: [:], positionDeltas: nil, queryOffset: 1, blockSize: 4, sampler: sampler
            )
            let logits = try XCTUnwrap(sampler.logits)
            XCTAssertEqual(logits.shape, [1, 3, 128])
            XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
            XCTAssertEqual(tokens.shape, [1, 3])
            XCTAssertTrue(MLX.all(tokens .>= 0).item(Bool.self))
            XCTAssertTrue(MLX.all(tokens .< 128).item(Bool.self))
        }
    }

    private final class CapturingSampler: LogitSampler {
        var logits: MLXArray?

        func sample(logits: MLXArray) -> MLXArray {
            self.logits = logits
            return argMax(logits, axis: -1)
        }
    }

    private func makeTarget() throws -> LagunaModel {
        let config = Data(
            #"{"model_type":"laguna","vocab_size":128,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"max_position_embeddings":128,"rms_norm_eps":0.000001,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":64,"shared_expert_intermediate_size":64,"layer_types":["sliding_attention"],"mlp_layer_types":["dense"],"sliding_window":8}"#
                .utf8
        )
        return LagunaModel(try JSONDecoder().decode(LagunaConfiguration.self, from: config))
    }
}

import Foundation
import GemmaActivationQuantizerCore
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MistralActivationScaleSearchCore
import WickModelSupport
import XCTest

final class GemmaAWSSCheckpointTests: XCTestCase {
    func testCompleteNativeCLIConversionReloadAndFailurePreservation() throws {
        let wick = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["WICK_TEST_EXECUTABLE"]
                ?? ".build/release/wick")
        let executable = wick.deletingLastPathComponent().appendingPathComponent(
            "wick-gemma-awss-quantize")
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw XCTSkip("Build wick-gemma-awss-quantize before this test")
        }
        try Device.withDefaultDevice(.gpu) {
            for moe in [false, true] {
                let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: root) }
                let source = root.appendingPathComponent("source")
                try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
                let text: [String: Any] = [
                    "model_type": "gemma4_text", "vocab_size": 32, "hidden_size": 128,
                    "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
                    "num_key_value_heads": 2, "num_global_key_value_heads": 2,
                    "head_dim": 32, "global_head_dim": 32, "sliding_window": 8,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
                    "attention_k_eq_v": false, "use_double_wide_mlp": false,
                    "enable_moe_block": moe, "num_experts": 4, "top_k_experts": 2,
                    "moe_intermediate_size": 64, "tie_word_embeddings": true,
                ]
                let textData = try JSONSerialization.data(withJSONObject: text)
                let config = try JSONSerialization.data(
                    withJSONObject: moe
                        ? ["model_type": "gemma4", "vocab_size": 32, "text_config": text] : text)
                try config.write(to: source.appendingPathComponent("config.json"))
                let model = Gemma4TextModel(
                    try JSONDecoder().decode(Gemma4TextConfiguration.self, from: textData))
                var originals = Dictionary(
                    uniqueKeysWithValues: model.parameters().flattened().map { name, tensor in
                        let values = (0..<tensor.size).map { Float(($0 * 7 + 3) % 41 - 20) / 41 }
                        return (name, MLXArray(values).reshaped(tensor.shape).asType(.bfloat16))
                    })
                if moe {
                    for layer in 0..<2 {
                        let prefix = "model.layers.\(layer).experts."
                        let gate = try XCTUnwrap(
                            originals.removeValue(forKey: prefix + "switch_glu.gate_proj.weight"))
                        let up = try XCTUnwrap(
                            originals.removeValue(forKey: prefix + "switch_glu.up_proj.weight"))
                        originals[prefix + "gate_up_proj"] = MLX.concatenated([gate, up], axis: -2)
                        originals[prefix + "down_proj"] = originals.removeValue(
                            forKey: prefix + "switch_glu.down_proj.weight")
                    }
                    originals = Dictionary(
                        uniqueKeysWithValues: originals.map { name, value in
                            (
                                name.replacingOccurrences(
                                    of: "model.", with: "model.language_model.", options: .anchored), value
                            )
                        })
                }
                try MLX.save(arrays: originals, url: source.appendingPathComponent("model.safetensors"))
                try JSONSerialization.data(withJSONObject: [
                    "weight_map": Dictionary(
                        uniqueKeysWithValues: originals.keys.map { ($0, "model.safetensors") })
                ])
                .write(to: source.appendingPathComponent("model.safetensors.index.json"))
                try Data("{}".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
                try Data("fixture sidecar\n".utf8).write(to: source.appendingPathComponent("notes.txt"))
                let identity = try GemmaActivationSourceIdentity.capture(source: source)
                let template = root.appendingPathComponent("template")
                var conversion = [
                    source.path, template.path, "--gemma-group-policy", "--calibration", "scale-search",
                    "--bounded-memory", "--cpu",
                ]
                if moe {
                    conversion += ["--q8-module", "*.router.proj", "--g128-module", "*.self_attn.o_proj"]
                }
                let templateRun = try run(wick, conversion, root: root)
                XCTAssertEqual(templateRun.0, 0, templateRun.1)
                guard templateRun.0 == 0 else { return }
                for (tokens, family) in [(Array(1...8), "fit"), (Array(9...16), "dev")] {
                    let collected = try Gemma4ActivationCollector.collect(
                        source: source, tokenSegments: [tokens], spoolParent: root, minimumExpertPositions: 2)
                    var moments = Dictionary(
                        uniqueKeysWithValues: collected.dense.map { ($0.path, $0.secondMoments) })
                    for expert in collected.experts { moments[expert.path] = expert.secondMoments }
                    let statistics = try GemmaActivationStatistics(
                        moments: moments,
                        expertCounts: Dictionary(
                            uniqueKeysWithValues: collected.experts.map { ($0.path, $0.expertPositionCounts) }),
                        provenance: GemmaActivationProvenance(
                            source: identity, corpus: Data(family.utf8), tokenSamples: [tokens],
                            tokenSegments: [tokens], sourceFamilies: [family]),
                        minimumExpertPositions: 2, expertsPerToken: moe ? 2 : 0)
                    try statistics.write(to: root.appendingPathComponent(family + ".safetensors"))
                }
                let output = root.appendingPathComponent("candidate")
                let args = [
                    source.path, template.path, root.appendingPathComponent("fit.safetensors").path,
                    root.appendingPathComponent("dev.safetensors").path, output.path,
                    "--minimum-expert-positions", "2", "--max-shard-gib", "0.0001",
                ]
                let provenanceURL = template.appendingPathComponent("scale-search-quantization.json")
                let provenanceData = try Data(contentsOf: provenanceURL)
                var recipe = try XCTUnwrap(
                    try JSONSerialization.jsonObject(with: provenanceData) as? [String: Any])
                for invalidMode: Any in [NSNull(), "symmetric"] {
                    recipe["mode"] = invalidMode
                    try JSONSerialization.data(withJSONObject: recipe).write(to: provenanceURL)
                    let rejected = try run(executable, args, root: root)
                    XCTAssertNotEqual(rejected.0, 0, rejected.1)
                    XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
                }
                recipe.removeValue(forKey: "mode")
                try JSONSerialization.data(withJSONObject: recipe).write(to: provenanceURL)
                let configURL = template.appendingPathComponent("config.json")
                let originalConfig = try Data(contentsOf: configURL)
                var invalidConfig = try XCTUnwrap(
                    try JSONSerialization.jsonObject(with: originalConfig) as? [String: Any])
                for aliasName in ["quantization", "quantization_config"] {
                    var policy = try XCTUnwrap(invalidConfig[aliasName] as? [String: Any])
                    policy["mode"] = "symmetric"
                    invalidConfig[aliasName] = policy
                }
                try JSONSerialization.data(withJSONObject: invalidConfig).write(to: configURL)
                let rejectedPolicy = try run(executable, args, root: root)
                XCTAssertNotEqual(rejectedPolicy.0, 0, rejectedPolicy.1)
                XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
                try originalConfig.write(to: configURL)
                let legacyProvenance = try Data(contentsOf: provenanceURL)
                let candidateRun = try run(executable, args, root: root)
                XCTAssertEqual(try Data(contentsOf: provenanceURL), legacyProvenance)
                XCTAssertEqual(candidateRun.0, 0, candidateRun.1)
                guard candidateRun.0 == 0 else { return }
                let baseline = try SelectiveSafetensorsReader(directory: template)
                let candidate = try SelectiveSafetensorsReader(directory: output)
                XCTAssertEqual(baseline.keys, candidate.keys)
                let reportData = try Data(
                    contentsOf: output.appendingPathComponent("gemma-awss-quantization.json"))
                let report = try XCTUnwrap(
                    try JSONSerialization.jsonObject(with: reportData) as? [String: Any])
                XCTAssertEqual(report["status"] as? String, "experimental_unbenchmarked_candidate")
                let selected = try XCTUnwrap(report["selectedModules"] as? [String])
                XCTAssertGreaterThan(selected.count, 7)
                XCTAssertFalse(
                    selected.contains {
                        $0.hasSuffix("router.proj") || $0.contains("embed_tokens") || $0.hasSuffix("lm_head")
                    })
                let selectedKeys = Set(
                    selected.flatMap { path in ["weight", "scales", "biases"].map { path + "." + $0 } })
                var changed = 0
                for name in baseline.keys {
                    let before = try baseline.read(name)
                    let after = try candidate.read(name)
                    XCTAssertEqual(before.shape, after.shape)
                    XCTAssertEqual(before.dtype, after.dtype)
                    if selectedKeys.contains(name) {
                        if before.asData().data != after.asData().data { changed += 1 }
                    } else {
                        XCTAssertEqual(before.asData().data, after.asData().data, name)
                    }
                }
                XCTAssertGreaterThan(changed, 0)
                XCTAssertEqual(
                    try Data(contentsOf: template.appendingPathComponent("config.json")),
                    try Data(contentsOf: output.appendingPathComponent("config.json")))
                XCTAssertEqual(
                    try Data(contentsOf: template.appendingPathComponent("notes.txt")),
                    try Data(contentsOf: output.appendingPathComponent("notes.txt")))
                let outputConfig = try Data(contentsOf: output.appendingPathComponent("config.json"))
                let reloaded: any LLMModel
                if moe {
                    reloaded = Gemma4Model(
                        try JSONDecoder().decode(Gemma4Configuration.self, from: outputConfig))
                } else {
                    reloaded = Gemma4TextModel(
                        try JSONDecoder().decode(Gemma4TextConfiguration.self, from: outputConfig))
                }
                let baseConfig = try JSONDecoder().decode(BaseConfiguration.self, from: outputConfig)
                try loadWeights(
                    modelDirectory: output, model: reloaded,
                    perLayerQuantization: baseConfig.perLayerQuantization)
                let logits = reloaded(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: nil)
                XCTAssertEqual(logits.shape, [1, 2, 32])
                XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
                let outputHash = try IndexedSafetensorsFingerprint.compute(directory: output)
                XCTAssertEqual(report["outputWeightsFingerprint"] as? String, outputHash)
                XCTAssertNotEqual(try run(executable, args, root: root).0, 0)
                var overlapping = args
                overlapping[3] = overlapping[2]
                XCTAssertNotEqual(try run(executable, overlapping + ["--overwrite"], root: root).0, 0)
                if moe {
                    XCTAssertFalse(selected.contains { $0.hasSuffix(".self_attn.o_proj") })
                    let router = "language_model.model.layers.0.router.proj"
                    XCTAssertNotEqual(
                        try run(executable, args + ["--overwrite", "--module", router], root: root).0, 0)
                }
                // Corrupt a selected template grid after a successful conversion.
                // This fails inside staging and must retain the previous output.
                let index = try XCTUnwrap(
                    try JSONSerialization.jsonObject(
                        with: Data(contentsOf: template.appendingPathComponent("model.safetensors.index.json")))
                        as? [String: Any])
                let map = try XCTUnwrap(index["weight_map"] as? [String: String])
                let corruptKey = try XCTUnwrap(selected.last) + ".scales"
                let corruptShard = template.appendingPathComponent(try XCTUnwrap(map[corruptKey]))
                var corrupted = try MLX.loadArrays(url: corruptShard)
                try MLX.checkedEval(Array(corrupted.values))
                corrupted[corruptKey] = try XCTUnwrap(corrupted[corruptKey]) + Float(1)
                try MLX.save(arrays: corrupted, url: corruptShard)
                let failure = try run(executable, args + ["--overwrite"], root: root)
                XCTAssertNotEqual(failure.0, 0)
                XCTAssertTrue(failure.1.contains("does not reproduce"), failure.1)
                XCTAssertEqual(try IndexedSafetensorsFingerprint.compute(directory: output), outputHash)
                XCTAssertEqual(
                    try Data(contentsOf: output.appendingPathComponent("gemma-awss-quantization.json")),
                    reportData)
                XCTAssertFalse(
                    try FileManager.default.contentsOfDirectory(atPath: root.path).contains {
                        $0.contains(".partial-") || $0.contains(".backup-")
                    })
                try identity.requireUnchanged()
                print(
                    "Gemma AWSS full-checkpoint fixture: moe=\(moe), selected=\(selected.count), changed_tensors=\(changed), native_reload=finite, rollback=preserved"
                )
            }
        }
    }

    private func run(_ executable: URL, _ args: [String], root: URL) throws -> (Int32, String) {
        let log = root.appendingPathComponent("command-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        let handle = try FileHandle(forWritingTo: log)
        defer { try? handle.close() }
        let process = Process()
        process.executableURL = executable
        process.arguments = args
        process.standardOutput = handle
        process.standardError = handle
        try process.run()
        process.waitUntilExit()
        return (process.terminationStatus, String(decoding: try Data(contentsOf: log), as: UTF8.self))
    }
}

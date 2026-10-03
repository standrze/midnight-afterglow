import Foundation
import MLX
import MLXLMCommon
import MLXNN
import WickModelSupport
import XCTest

final class TalkieConversionTests: XCTestCase {
    func testStandardQ4FoldsCanonicalBF16HeadBeforeQuantization() throws {
        try verifyConversion(bits: 4, calibration: .standard)
    }

    func testStandardQ8FoldsCanonicalBF16HeadBeforeQuantization() throws {
        try verifyConversion(bits: 8, calibration: .standard)
    }

    func testScaleSearchPreservesNativeTalkieLayoutAndReloads() throws {
        try verifyConversion(bits: 4, calibration: .q4AffineScaleSearch)
    }

    func testMixedPrecisionKeepsSourceModulePolicyOnReload() throws {
        try verifyConversion(bits: 4, calibration: .standard, q8Head: true)
    }

    private func verifyConversion(
        bits: Int, calibration: ModelConversionQuantizationCalibration, q8Head: Bool = false
    ) throws {
        try Device.withDefaultDevice(.cpu) {
            let fileManager = FileManager.default
            let root = fileManager.temporaryDirectory.appendingPathComponent(
                "talkie-conversion-\(UUID().uuidString)")
            let source = root.appendingPathComponent("source")
            let output = root.appendingPathComponent("output")
            try fileManager.createDirectory(at: source, withIntermediateDirectories: true)
            defer { try? fileManager.removeItem(at: root) }

            let fixture = try makeFixture(at: source)
            let configuration = fixture.configuration
            let sourceWeights = fixture.weights
            let sidecars = fixture.sidecars
            let head = fixture.head
            let gain = fixture.gain
            let skipKey = fixture.skipKey

            let result = try MLXLMCommon.convert(
                modelDirectory: source,
                model: TalkieModel(configuration, fuseProjections: false),
                to: output,
                options: ModelConversionOptions(
                    bits: bits, groupSize: 64, mode: .affine, calibration: calibration,
                    maxShardSize: 16 * 1_024 * 1_024,
                    quantizationPredicate: { path, _ in
                        if q8Head && path == "lm_head" {
                            return .quantize(.init(bits: 8, groupSize: 64, mode: .affine, calibration: .standard))
                        }
                        return .quantize()
                    }))

            var outputWeights = [String: MLXArray]()
            for url in result.weightsURLs {
                outputWeights.merge(try loadArrays(url: url, stream: .cpu), uniquingKeysWith: { _, new in new })
            }
            XCTAssertNil(outputWeights["lm_head"])
            XCTAssertNil(outputWeights["lm_head_gain.w_g"])
            XCTAssertNotNil(outputWeights["lm_head.scales"])
            XCTAssertNotNil(outputWeights["model.blocks.0.attn.attn_query.scales"])
            XCTAssertNotNil(outputWeights["model.blocks.0.mlp.mlp_gate.scales"])
            XCTAssertNil(outputWeights["model.blocks.0.attn.attn_qkv.weight"])
            XCTAssertNil(outputWeights["model.blocks.0.mlp.mlp_gate_up.weight"])
            XCTAssertTrue(
                MLX.arrayEqual(
                    try XCTUnwrap(outputWeights[skipKey]),
                    try XCTUnwrap(sourceWeights[skipKey])
                ).item(Bool.self))
            for (filename, expected) in sidecars {
                XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent(filename)), expected, filename)
            }

            if calibration == .standard {
                let expected = MLX.quantized(
                    head * gain.asType(head.dtype), groupSize: 64,
                    bits: q8Head ? 8 : bits, mode: .affine)
                XCTAssertTrue(
                    MLX.arrayEqual(try XCTUnwrap(outputWeights["lm_head.weight"]), expected.wq).item(Bool.self))
                XCTAssertTrue(
                    MLX.arrayEqual(try XCTUnwrap(outputWeights["lm_head.scales"]), expected.scales).item(Bool.self))
                XCTAssertTrue(
                    MLX.arrayEqual(try XCTUnwrap(outputWeights["lm_head.biases"]), try XCTUnwrap(expected.biases)).item(
                        Bool.self))
            }

            let convertedData = try Data(contentsOf: output.appendingPathComponent("config.json"))
            let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: convertedData)
            let quantization = try XCTUnwrap(base.perLayerQuantization)
            XCTAssertEqual(quantization.quantization?.bits, bits)
            XCTAssertEqual(quantization.quantization?.groupSize, 64)
            // Uniform recipes must stay uniform so runtime projection fusion remains
            // available; only a genuinely different head precision needs an override.
            XCTAssertEqual(quantization.perLayerQuantization.count, q8Head ? 1 : 0)
            XCTAssertEqual(quantization.quantization(layer: "lm_head")?.bits, q8Head ? 8 : bits)
            let converted = try JSONDecoder().decode(TalkieConfiguration.self, from: convertedData)
            let input = MLXArray([Int32(3), 1, 4])[.newAxis]
            for fused in [false, true] {
                let reloaded = TalkieModel(converted, fuseProjections: fused)
                try loadWeights(modelDirectory: output, model: reloaded, perLayerQuantization: quantization)
                let logits = reloaded(input, cache: nil)
                XCTAssertEqual(logits.shape, [1, 3, 128])
                XCTAssertTrue(MLX.max(MLX.abs(logits)).item(Float.self).isFinite)
                if q8Head { XCTAssertFalse(reloaded.usesFusedProjections) }
            }
        }
    }

    func testCLIProvenanceAndOverwriteFailurePreserveCompletedCheckpoint() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let executable = environment["WICK_TEST_EXECUTABLE"] ?? environment["FACET_TEST_EXECUTABLE"],
            !executable.isEmpty
        else {
            throw XCTSkip("Set WICK_TEST_EXECUTABLE to the built wick executable to run the CLI conversion regression.")
        }
        try Device.withDefaultDevice(.cpu) {
            let manager = FileManager.default
            let root = manager.temporaryDirectory.appendingPathComponent("wick-cli-conversion-\(UUID())")
            let source = root.appendingPathComponent("source")
            let output = root.appendingPathComponent("output")
            try manager.createDirectory(at: source, withIntermediateDirectories: true)
            defer { try? manager.removeItem(at: root) }
            let fixture = try makeFixture(at: source)
            let arguments = [source.path, output.path, "--standard-q4", "--cpu"]

            let converted = try runCLI(executable: executable, arguments: arguments)
            XCTAssertEqual(converted.status, 0, converted.output)
            XCTAssertEqual(converted.reason, .exit, converted.output)
            XCTAssertTrue(converted.output.contains("Created \(output.path)"), converted.output)
            XCTAssertFalse(converted.output.contains(".partial-"), converted.output)
            let provenanceData = try Data(contentsOf: output.appendingPathComponent("standard-q4-quantization.json"))
            let provenance = try XCTUnwrap(try JSONSerialization.jsonObject(with: provenanceData) as? [String: Any])
            XCTAssertEqual(provenance["tool"] as? String, "midnight-afterglow")
            let version = try XCTUnwrap(provenance["tool_version"] as? String)
            XCTAssertNotNil(version.range(of: #"^\d+\.\d+\.\d+(?:[-+].*)?$"#, options: .regularExpression))
            XCTAssertFalse(String(decoding: provenanceData, as: UTF8.self).contains(".partial-"))
            let shards = try XCTUnwrap(provenance["output_shards"] as? [String])
            XCTAssertFalse(shards.isEmpty)
            for shard in shards {
                XCTAssertEqual(shard, URL(fileURLWithPath: shard).lastPathComponent)
                XCTAssertTrue(manager.fileExists(atPath: output.appendingPathComponent(shard).path))
            }
            let previousFiles = try checkpointFiles(at: output)

            // A valid safetensors file with one required matrix omitted passes input
            // preflight and fails while loading weights after staging has started.
            var incompleteWeights = fixture.weights
            _ = try XCTUnwrap(incompleteWeights.removeValue(forKey: "model.blocks.0.attn.attn_query.weight"))
            try save(
                arrays: incompleteWeights, metadata: ["format": "pt"],
                url: source.appendingPathComponent("model.safetensors"), stream: .cpu)
            let failed = try runCLI(executable: executable, arguments: arguments + ["--overwrite"])
            XCTAssertNotEqual(failed.status, 0, failed.output)
            XCTAssertEqual(failed.reason, .exit, failed.output)
            XCTAssertTrue(failed.output.contains("[loadingWeights]"), failed.output)
            XCTAssertFalse(failed.output.contains("Created \(output.path)"), failed.output)
            XCTAssertEqual(try checkpointFiles(at: output), previousFiles)
            XCTAssertEqual(Set(try manager.contentsOfDirectory(atPath: root.path)), ["source", "output"])
        }
    }

    private struct Fixture {
        let configuration: TalkieConfiguration
        let weights: [String: MLXArray]
        let sidecars: [String: Data]
        let head: MLXArray
        let gain: MLXArray
        let skipKey: String
    }

    private func makeFixture(at source: URL) throws -> Fixture {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/TalkieTiny/config.json")
        let configData = try Data(contentsOf: fixture)
        try configData.write(to: source.appendingPathComponent("config.json"))
        let template =
            "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
        let sidecars = [
            "chat_template.jinja": Data(template.utf8),
            "tokenizer_config.json": Data(
                #"{"tokenizer_class":"TokenizersBackend","bos_token":null,"model_max_length":2048}"#.utf8),
            "tokenizer.json": Data(#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]}}"#.utf8),
            "generation_config.json": Data(#"{"eos_token_id":[126,127],"do_sample":true,"temperature":0.7}"#.utf8),
        ]
        for (filename, data) in sidecars {
            try data.write(to: source.appendingPathComponent(filename))
        }
        let configuration = try JSONDecoder().decode(TalkieConfiguration.self, from: configData)
        let sourceModel = TalkieModel(configuration, fuseProjections: false)
        var sourceWeights = Dictionary(
            uniqueKeysWithValues:
                sourceModel.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) })
        let head = try XCTUnwrap(sourceWeights.removeValue(forKey: "lm_head.weight"))
        let gain = MLXArray([Float(1.375)]).asType(.bfloat16)
        sourceWeights["lm_head"] = head
        sourceWeights["lm_head_gain.w_g"] = gain
        // Keep a nontrivial scalar gain as evidence that only the output weight gain
        // is folded; activation gains must survive checkpoint conversion unchanged.
        let skipKey = "model.blocks.0.embed_skip.a_g"
        sourceWeights[skipKey] = MLXArray([Float(0.125)]).asType(.bfloat16)
        try save(
            arrays: sourceWeights, metadata: ["format": "pt"],
            url: source.appendingPathComponent("model.safetensors"), stream: .cpu)

        return Fixture(
            configuration: configuration, weights: sourceWeights, sidecars: sidecars,
            head: head, gain: gain, skipKey: skipKey)
    }

    private func runCLI(executable: String, arguments: [String]) throws -> (
        status: Int32, reason: Process.TerminationReason, output: String
    ) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, process.terminationReason, String(decoding: data, as: UTF8.self))
    }

    private func checkpointFiles(at directory: URL) throws -> [String: Data] {
        try Dictionary(
            uniqueKeysWithValues: FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil
            ).map {
                ($0.lastPathComponent, try Data(contentsOf: $0))
            })
    }
}

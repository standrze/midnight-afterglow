import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

final class QuantizationFormatsTests: XCTestCase {
    func testStandardFormatsConvertAndReloadFromCLI() throws {
        try verifyStandardFormats(useCPU: true)
    }

    func testStandardFormatsConvertAndReloadOnMetal() throws {
        try verifyStandardFormats(useCPU: false)
    }

    private func verifyStandardFormats(useCPU: Bool) throws {
        let executable = try executablePath()
        try Device.withDefaultDevice(useCPU ? .cpu : .gpu) {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let source = root.appendingPathComponent("source")
            let fixture = try makeMistralFixture(at: source)
            let recipes: [(String, QuantizationMode, Int, Int, [String])] = [
                ("affine", .affine, 4, 64, ["--mode", "affine"]),
                ("mxfp4", .mxfp4, 4, 32, ["--mode", "mxfp4"]),
                ("mxfp8", .mxfp8, 8, 32, ["--mode", "mxfp8"]),
                ("nvfp4", .nvfp4, 4, 16, ["--mode", "nvfp4"]),
                ("affine2", .affine, 2, 32, ["--bits", "2", "--group-size", "32"]),
                ("affine3", .affine, 3, 128, ["--bits", "3", "--group-size", "128"]),
                ("affine5", .affine, 5, 32, ["--bits", "5", "--group-size", "32"]),
                ("affine6", .affine, 6, 128, ["--bits", "6", "--group-size", "128"]),
                ("affine8", .affine, 8, 128, ["--bits", "8", "--group-size", "128"]),
            ]

            for (name, mode, bits, groupSize, options) in recipes {
                let output = root.appendingPathComponent(name)
                let converted = try runCLI(
                    executable: executable,
                    arguments: ["quantize", source.path, output.path] + (useCPU ? ["--cpu"] : []) + options
                )
                try requireSuccess(converted)

                let config = try readJSON(output.appendingPathComponent("config.json"))
                let quantization = try XCTUnwrap(config["quantization"] as? [String: Any])
                let modeName = mode == .affine ? "affine" : name
                XCTAssertEqual(quantization["mode"] as? String, modeName, name)
                XCTAssertEqual(quantization["bits"] as? Int, bits, name)
                XCTAssertEqual(quantization["group_size"] as? Int, groupSize, name)

                let weights = try readWeights(at: output)
                let path = "model.layers.0.self_attn.q_proj"
                let original = try XCTUnwrap(fixture.weights[path + ".weight"])
                try assertStandardPacking(
                    weights: weights, path: path, original: original,
                    mode: mode, bits: bits, groupSize: groupSize
                )
                let model = LlamaModel(fixture.configuration)
                try reloadAndCheckInference(model: model, directory: output)

                let legacyName =
                    mode == .affine && bits == 4 && groupSize == 64
                    ? "standard-q4-quantization.json" : "quantization.json"
                let provenanceURL = output.appendingPathComponent(legacyName)
                let provenance = try readJSON(provenanceURL)
                XCTAssertEqual(provenance["mode"] as? String, modeName, name)
                XCTAssertEqual(provenance["bits"] as? Int, bits, name)
                XCTAssertEqual(provenance["group_size"] as? Int, groupSize, name)
                let standardModules = try XCTUnwrap(provenance["standard_modules"] as? [String])
                let q8Modules = try XCTUnwrap(provenance["q8_modules"] as? [String])
                XCTAssertTrue(standardModules.contains(path) || q8Modules.contains(path), name)
                XCTAssertTrue(q8Modules.isEmpty, name)
                XCTAssertEqual(provenance["q4_scale_search_modules"] as? [String], [], name)
            }
        }
    }

    func testMXFP4MixtralRetainsAffineQ8RoutersAndReloads() throws {
        let executable = try executablePath()
        try Device.withDefaultDevice(.cpu) {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let source = root.appendingPathComponent("source")
            let output = root.appendingPathComponent("output")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            let data = Data(
                #"{"model_type":"mixtral","vocab_size":128,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"num_local_experts":4,"num_experts_per_tok":2,"rms_norm_eps":0.00001,"rope_theta":1000000,"tie_word_embeddings":false}"#
                    .utf8
            )
            try data.write(to: source.appendingPathComponent("config.json"))
            let configuration = try JSONDecoder().decode(MixtralConfiguration.self, from: data)
            let sourceModel = MixtralModel(configuration)
            let originals = deterministicWeights(model: sourceModel)
            try save(
                arrays: originals, url: source.appendingPathComponent("model.safetensors"), stream: .cpu
            )

            let result = try runCLI(
                executable: executable,
                arguments: ["quantize", source.path, output.path, "--mode", "mxfp4", "--cpu"]
            )
            try requireSuccess(result)
            let config = try readJSON(output.appendingPathComponent("config.json"))
            let quantization = try XCTUnwrap(config["quantization"] as? [String: Any])
            XCTAssertEqual(quantization["mode"] as? String, "mxfp4")
            XCTAssertEqual(quantization["bits"] as? Int, 4)
            XCTAssertEqual(quantization["group_size"] as? Int, 32)
            let router = "model.layers.0.block_sparse_moe.gate"
            let override = try XCTUnwrap(quantization[router] as? [String: Any])
            XCTAssertEqual(override["mode"] as? String, "affine")
            XCTAssertEqual(override["bits"] as? Int, 8)
            XCTAssertEqual(override["group_size"] as? Int, 64)

            let weights = try readWeights(at: output)
            try assertStandardPacking(
                weights: weights, path: router, original: try XCTUnwrap(originals[router + ".weight"]),
                mode: .affine, bits: 8, groupSize: 64
            )
            let expert = "model.layers.0.block_sparse_moe.switch_mlp.gate_proj"
            XCTAssertNotNil(weights[expert + ".weight"])
            XCTAssertNotNil(weights[expert + ".scales"])
            XCTAssertNil(weights[expert + ".biases"])
            let model = MixtralModel(configuration)
            try reloadAndCheckInference(model: model, directory: output)

            let provenance = try readJSON(output.appendingPathComponent("quantization.json"))
            XCTAssertEqual(provenance["mode"] as? String, "mxfp4")
            XCTAssertEqual(provenance["q8_modules"] as? [String], [router])
            XCTAssertEqual(provenance["mandatory_q8_modules"] as? [String], [router])
            XCTAssertTrue(try XCTUnwrap(provenance["standard_modules"] as? [String]).contains(expert))
            XCTAssertEqual(provenance["q4_scale_search_modules"] as? [String], [])
        }
    }

    func testQ4G32ScaleSearchConvertsAndReloadsOnCPUAndMetal() throws {
        let executable = try executablePath()
        for device in [Device.cpu, Device.gpu] {
            try Device.withDefaultDevice(device) {
                let root = try temporaryDirectory()
                defer { try? FileManager.default.removeItem(at: root) }
                let source = root.appendingPathComponent("source")
                let fixture = try makeMistralFixture(at: source)
                let output = root.appendingPathComponent("q4-g32-scale-search")
                let result = try runCLI(
                    executable: executable,
                    arguments: [
                        "quantize", source.path, output.path, "--bits", "4", "--group-size", "32",
                        "--calibration", "scale-search",
                    ] + (device == .cpu ? ["--cpu"] : []))
                try requireSuccess(result)
                let config = try readJSON(output.appendingPathComponent("config.json"))
                let quantization = try XCTUnwrap(config["quantization"] as? [String: Any])
                XCTAssertEqual(quantization["bits"] as? Int, 4)
                XCTAssertEqual(quantization["group_size"] as? Int, 32)
                let weights = try readWeights(at: output)
                let path = "model.layers.0.self_attn.q_proj"
                let original = try XCTUnwrap(fixture.weights[path + ".weight"])
                let expected = q4AffineScaleSearchQuantized(original, groupSize: 32)
                for (suffix, value) in [
                    (".weight", expected.weight), (".scales", expected.scales),
                    (".biases", expected.biases),
                ] {
                    XCTAssertTrue(MLX.all(try XCTUnwrap(weights[path + suffix]) .== value).item(Bool.self))
                }
                let model = LlamaModel(fixture.configuration)
                try reloadAndCheckInference(model: model, directory: output)
                XCTAssertTrue(
                    FileManager.default.fileExists(atPath: output.appendingPathComponent("quantization.json").path))
            }
        }
    }

    func testHelpFormatsAndExplicitQuantization() throws {
        let executable = try executablePath()
        let rootHelp = try runCLI(executable: executable, arguments: ["--help"])
        try requireSuccess(rootHelp)
        XCTAssertTrue(rootHelp.output.contains("quantize"), rootHelp.output)
        XCTAssertTrue(rootHelp.output.contains("formats"), rootHelp.output)
        let help = try runCLI(executable: executable, arguments: ["quantize", "--help"])
        try requireSuccess(help)
        for option in ["--mode", "--bits", "--group-size", "--calibration", "--standard-q4", "--standard-q8"] {
            XCTAssertTrue(help.output.contains(option), help.output)
        }
        let formats = try runCLI(executable: executable, arguments: ["formats"])
        try requireSuccess(formats)
        for mode in ["affine", "mxfp4", "mxfp8", "nvfp4"] {
            XCTAssertTrue(formats.output.contains(mode), formats.output)
        }

        try Device.withDefaultDevice(.cpu) {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let source = root.appendingPathComponent("source")
            _ = try makeMistralFixture(at: source)
            let destination = root.appendingPathComponent("output")
            let arguments = [source.path, destination.path, "--dry-run"]
            let explicit = try runCLI(executable: executable, arguments: ["quantize"] + arguments)
            let selected = try runCLI(
                executable: executable,
                arguments: ["quantize"] + arguments + ["--calibration", "scale-search"]
            )
            try requireSuccess(explicit)
            try requireSuccess(selected)
            XCTAssertEqual(explicit.output, selected.output)
            XCTAssertTrue(explicit.output.contains("ScaleSearch"), explicit.output)
            XCTAssertFalse(FileManager.default.fileExists(atPath: destination.path))

            for flag in ["--standard-q4", "--standard-q8"] {
                let compatible = try runCLI(executable: executable, arguments: arguments + [flag])
                try requireSuccess(compatible)
            }
        }
    }

    func testInvalidRecipesFailBeforeReadingSource() throws {
        let executable = try executablePath()
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("missing-source")
        let output = root.appendingPathComponent("output")
        let recipes = [
            ["--mode", "unknown"],
            ["--bits", "7"],
            ["--mode", "mxfp4", "--bits", "8"],
            ["--mode", "mxfp8", "--bits", "4"],
            ["--mode", "nvfp4", "--group-size", "32"],
            ["--mode", "mxfp4", "--group-size", "64"],
            ["--mode", "affine", "--group-size", "16"],
            ["--mode", "mxfp4", "--calibration", "scale-search"],
            ["--bits", "8", "--calibration", "scale-search"],
            ["--bits", "5", "--group-size", "32", "--calibration", "scale-search"],
            ["--standard-q4", "--standard-q8"],
            ["--standard-q8", "--group-size", "128"],
            ["--standard-q4", "--mode", "mxfp4"],
            ["--standard-q8", "--bits", "4"],
            ["--standard-q4", "--calibration", "scale-search"],
            ["--mode", "mxfp4", "--template", "/missing-template"],
            ["--mode", "mxfp4", "--gemma-group-policy"],
        ]

        for options in recipes {
            let result = try runCLI(
                executable: executable,
                arguments: ["quantize", source.path, output.path] + options
            )
            XCTAssertNotEqual(result.status, 0, options.joined(separator: " "))
            XCTAssertEqual(result.reason, .exit, result.output)
            XCTAssertFalse(result.output.contains("source directory does not exist"), result.output)
            XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        }
    }

    private struct MistralFixture {
        let configuration: LlamaConfiguration
        let weights: [String: MLXArray]
    }

    private struct CLIResult {
        let status: Int32
        let reason: Process.TerminationReason
        let output: String
    }

    private func executablePath() throws -> String {
        let environment = ProcessInfo.processInfo.environment
        guard let executable = environment["AFTERGLOW_TEST_EXECUTABLE"],
            !executable.isEmpty
        else {
            throw XCTSkip(
                "Set AFTERGLOW_TEST_EXECUTABLE to the built midnight-afterglow executable for CLI format tests.")
        }
        return executable
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("wick-formats-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func makeMistralFixture(at directory: URL) throws -> MistralFixture {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let data = Data(
            #"{"model_type":"mistral","vocab_size":128,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":0.00001,"max_position_embeddings":256,"rope_theta":10000,"tie_word_embeddings":false,"attention_bias":false,"mlp_bias":false}"#
                .utf8
        )
        try data.write(to: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(LlamaConfiguration.self, from: data)
        let model = LlamaModel(configuration)
        let weights = deterministicWeights(model: model)
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"), stream: .cpu)
        return MistralFixture(configuration: configuration, weights: weights)
    }

    private func deterministicWeights(model: Module) -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { name, weight in
                guard weight.ndim >= 2 else { return (name, weight) }
                let values = (0..<weight.size).map { Float(($0 * 17) % 101 - 50) / 500 }
                return (name, MLXArray(values).reshaped(weight.shape))
            })
    }

    private func readJSON(_ url: URL) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    private func readWeights(at directory: URL) throws -> [String: MLXArray] {
        let urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        XCTAssertFalse(urls.isEmpty)
        var weights = [String: MLXArray]()
        for url in urls {
            weights.merge(try loadArrays(url: url, stream: .cpu), uniquingKeysWith: { _, new in new })
        }
        return weights
    }

    private func assertStandardPacking(
        weights: [String: MLXArray], path: String, original: MLXArray,
        mode: QuantizationMode, bits: Int, groupSize: Int
    ) throws {
        let expected = MLX.quantized(original, groupSize: groupSize, bits: bits, mode: mode)
        let packed = try XCTUnwrap(weights[path + ".weight"])
        let scales = try XCTUnwrap(weights[path + ".scales"])
        XCTAssertEqual(packed.dtype, .uint32, path)
        XCTAssertEqual(packed.shape, Array(original.shape.dropLast()) + [original.dim(-1) * bits / 32], path)
        XCTAssertEqual(scales.shape, Array(original.shape.dropLast()) + [original.dim(-1) / groupSize], path)
        XCTAssertTrue(MLX.arrayEqual(packed, expected.wq).item(Bool.self), path)
        XCTAssertTrue(MLX.arrayEqual(scales, expected.scales).item(Bool.self), path)
        if let expectedBiases = expected.biases {
            let biases = try XCTUnwrap(weights[path + ".biases"])
            XCTAssertTrue(MLX.arrayEqual(biases, expectedBiases).item(Bool.self), path)
        } else {
            XCTAssertNil(weights[path + ".biases"], path)
        }
    }

    private func reloadAndCheckInference(model: any LLMModel, directory: URL) throws {
        let base = try JSONDecoder.json5().decode(
            BaseConfiguration.self, from: Data(contentsOf: directory.appendingPathComponent("config.json"))
        )
        try loadWeights(modelDirectory: directory, model: model, perLayerQuantization: base.perLayerQuantization)
        let input = MLXArray([Int32(1), 3, 7])[.newAxis]
        let logits = model(input, cache: nil)
        XCTAssertEqual(logits.shape, [1, 3, 128])
        XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
    }

    private func runCLI(executable: String, arguments: [String]) throws -> CLIResult {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments =
            arguments.first.map { ["quantize", "formats", "--help"].contains($0) } == true
            ? arguments : ["quantize"] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return CLIResult(
            status: process.terminationStatus, reason: process.terminationReason,
            output: String(decoding: data, as: UTF8.self)
        )
    }

    private func requireSuccess(_ result: CLIResult) throws {
        XCTAssertEqual(result.reason, .exit, result.output)
        XCTAssertEqual(result.status, 0, result.output)
        guard result.reason == .exit, result.status == 0 else {
            throw NSError(
                domain: "WickFormatTests", code: Int(result.status),
                userInfo: [
                    NSLocalizedDescriptionKey: result.output
                ])
        }
    }
}

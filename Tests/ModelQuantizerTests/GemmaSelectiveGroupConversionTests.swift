import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import QuantizerSupport
import WickModelSupport
import XCTest

final class GemmaSelectiveGroupConversionTests: XCTestCase {
    func testQ8TiedEmbeddingAndNativeHeadReloadWithTheSameGrid() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            let source = directory.appendingPathComponent("source")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = Data(
                #"{"model_type":"gemma3_text","vocab_size":32,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":6,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"sliding_window":128,"tie_word_embeddings":true}"#
                    .utf8)
            try config.write(to: source.appendingPathComponent("config.json"))
            let configuration = try JSONDecoder().decode(Gemma3TextConfiguration.self, from: config)
            let native = Gemma3TextModel(configuration)
            var arrays = Dictionary(
                uniqueKeysWithValues: native.parameters().flattened().map { name, weight in
                    let values = (0..<weight.size).map { Float($0 % 37 - 18) / 41 }
                    return (name, MLXArray(values).reshaped(weight.shape).asType(.bfloat16))
                })
            arrays["lm_head.weight"] = arrays["model.embed_tokens.weight"]
            try save(arrays: arrays, url: source.appendingPathComponent("model.safetensors"))
            var control = [String: MLXArray]()
            for q8Head in [false, true] {
                let model = Gemma3TiedHeadConversionModel(Gemma3TextModel(configuration))
                let leaves = model.leafModules().flattened().filter { $0.1 is Quantizable }
                let modules = Dictionary(
                    uniqueKeysWithValues: leaves.map { name, module in
                        let shape = module.parameters().flattened().first { $0.0 == "weight" }!.1.shape
                        return (
                            name,
                            GemmaGroupSizePolicy.Module(
                                shape: shape, kind: module is Embedding ? .embedding : .linear)
                        )
                    })
                let policy = try GemmaGroupSizePolicy(
                    sourceConfiguration: config, modules: modules, selectedModules: [],
                    q8Modules: q8Head ? ["model.embed_tokens"] : [], omittedTiedHead: true)
                var options = ModelConversionOptions(
                    bits: 4, groupSize: 64, calibration: .q4AffineScaleSearch,
                    quantizationPredicate: { path, _ in
                        if q8Head && path == "model.embed_tokens" {
                            return .quantize(.init(bits: 8, groupSize: 64, calibration: .standard))
                        }
                        return .quantize()
                    })
                options.boundedMemory = true
                let destination = directory.appendingPathComponent(q8Head ? "q8-head" : "q4-head")
                let result = try MLXLMCommon.convert(
                    modelDirectory: source, model: model, to: destination, options: options)
                let configurationURL = destination.appendingPathComponent("config.json")
                let original = try Data(contentsOf: configurationURL)
                if q8Head { XCTAssertThrowsError(try policy.validateOutputConfiguration(original)) }
                let finalized = try policy.finalizedOutputConfiguration(original)
                if !q8Head { XCTAssertEqual(finalized, original) }
                try finalized.write(to: configurationURL, options: .atomic)
                try policy.validateOutputWeights(result.weightsURLs)
                var exported = [String: MLXArray]()
                for shard in result.weightsURLs { exported.merge(try loadArrays(url: shard)) { _, new in new } }
                XCTAssertFalse(exported.keys.contains { $0.hasPrefix("lm_head.") })
                if q8Head {
                    XCTAssertEqual(Set(exported.keys), Set(control.keys))
                    for (name, tensor) in control where !name.hasPrefix("model.embed_tokens.") {
                        let actual = try XCTUnwrap(exported[name])
                        XCTAssertEqual(actual.shape, tensor.shape, name)
                        XCTAssertTrue(MLX.all(actual .== tensor).item(Bool.self), name)
                    }
                } else {
                    control = exported
                }
                for (name, tensor) in arrays where name.contains("norm") {
                    let actual = try XCTUnwrap(exported[name])
                    XCTAssertTrue(MLX.all(actual .== tensor).item(Bool.self), name)
                }
                let restored = Gemma3TextModel(configuration)
                let base = try JSONDecoder().decode(BaseConfiguration.self, from: finalized)
                try loadWeights(
                    modelDirectory: destination, model: restored, perLayerQuantization: base.perLayerQuantization)
                let restoredModules = Dictionary(uniqueKeysWithValues: restored.leafModules().flattened())
                let head = try XCTUnwrap(restoredModules["lm_head"] as? QuantizedLinear)
                let embedding = try XCTUnwrap(restoredModules["model.embed_tokens"] as? QuantizedEmbedding)
                XCTAssertEqual(head.bits, q8Head ? 8 : 4)
                XCTAssertEqual(embedding.bits, head.bits)
                XCTAssertEqual(head.groupSize, 64)
                XCTAssertEqual(embedding.groupSize, 64)
                XCTAssertEqual(head.weight.shape, [32, q8Head ? 32 : 16])
                let restoredWeights = Dictionary(uniqueKeysWithValues: restored.parameters().flattened())
                for suffix in ["weight", "scales", "biases"] {
                    let headTensor = try XCTUnwrap(restoredWeights["lm_head." + suffix])
                    let embeddingTensor = try XCTUnwrap(restoredWeights["model.embed_tokens." + suffix])
                    XCTAssertTrue(MLX.all(headTensor .== embeddingTensor).item(Bool.self), suffix)
                }
                let hidden = MLX.ones([1, 128], dtype: .bfloat16)
                XCTAssertTrue(MLX.all(head(hidden) .== embedding.asLinear(hidden)).item(Bool.self))
                let logits = restored(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: nil)
                XCTAssertEqual(logits.shape, [1, 2, 32])
                XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
            }
        }
    }

    private final class ExpertFixture: Module, BaseLanguageModel {
        @ModuleInfo var experts: SwitchLinear

        override init() {
            _experts.wrappedValue = SwitchLinear(inputDims: 128, outputDims: 64, numExperts: 2, bias: false)
            super.init()
        }
    }

    func testSearchedSwitchLinearUsesTheRequestedG128Grid() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            let source = directory.appendingPathComponent("source")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let config = Data(#"{"model_type":"gemma4_text"}"#.utf8)
            try config.write(to: source.appendingPathComponent("config.json"))
            let values = (0..<(2 * 64 * 128)).map { Float($0 % 53 - 26) / 47 }
            let weight = MLXArray(values).reshaped(2, 64, 128).asType(.bfloat16)
            try save(arrays: ["experts.weight": weight], url: source.appendingPathComponent("model.safetensors"))
            let policy = try GemmaGroupSizePolicy(
                sourceConfiguration: config,
                modules: ["experts": .init(shape: [2, 64, 128], kind: .switchLinear)],
                selectedModules: ["experts"])
            var options = ModelConversionOptions(
                bits: 4, groupSize: 64, calibration: .q4AffineScaleSearch,
                quantizationPredicate: { path, _ in
                    path == "experts"
                        ? .quantize(.init(bits: 4, groupSize: 128, calibration: .q4AffineScaleSearch)) : .quantize()
                })
            options.boundedMemory = true
            let result = try MLXLMCommon.convert(
                modelDirectory: source, model: ExpertFixture(), to: directory.appendingPathComponent("converted"),
                options: options)
            try policy.validateOutputConfiguration(
                Data(contentsOf: result.outputDirectory.appendingPathComponent("config.json")))
            try policy.validateOutputWeights(result.weightsURLs)
            let actual = try loadArrays(url: XCTUnwrap(result.weightsURLs.first))
            let expected = q4AffineScaleSearchQuantized(weight, groupSize: 128, rowBatchSize: 512)
            for (suffix, value) in [
                ("weight", expected.weight), ("scales", expected.scales), ("biases", expected.biases),
            ] {
                let tensor = try XCTUnwrap(actual["experts." + suffix])
                XCTAssertEqual(tensor.shape, value.shape)
                XCTAssertTrue(MLX.all(tensor .== value).item(Bool.self), suffix)
            }
        }
    }

    func testOrdinaryAndSearchedMixedGroupsRoundTripOnCPU() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let source = directory.appendingPathComponent("source")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            let config = Data(
                #"{"model_type":"gemma3_text","vocab_size":32,"hidden_size":128,"intermediate_size":256,"num_hidden_layers":6,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,"sliding_window":128,"tie_word_embeddings":true}"#
                    .utf8)
            try config.write(to: source.appendingPathComponent("config.json"))
            let configuration = try JSONDecoder().decode(Gemma3TextConfiguration.self, from: config)
            let native = Gemma3TextModel(configuration)
            var arrays = Dictionary(
                uniqueKeysWithValues: native.parameters().flattened().map { name, weight in
                    let values = (0..<weight.size).map { Float($0 % 31 - 15) / 32 }
                    return (name, MLXArray(values).reshaped(weight.shape).asType(.bfloat16))
                })
            arrays["lm_head.weight"] = arrays["model.embed_tokens.weight"]
            let sourceWeights = source.appendingPathComponent("model.safetensors")
            try save(arrays: arrays, url: sourceWeights)
            XCTAssertTrue(
                try GemmaGroupSizePolicy.preservesGemma3TiedHead(
                    sourceConfiguration: config, shards: [sourceWeights]))

            for calibration in [ModelConversionQuantizationCalibration.standard, .q4AffineScaleSearch] {
                var control = [String: MLXArray]()
                for useG128 in [false, true] {
                    let model = Gemma3TiedHeadConversionModel(Gemma3TextModel(configuration))
                    let leaves = model.leafModules().flattened().filter { $0.1 is Quantizable }
                    let modules = Dictionary(
                        uniqueKeysWithValues: leaves.map { name, module in
                            let shape = module.parameters().flattened().first { $0.0 == "weight" }!.1.shape
                            return (
                                name,
                                GemmaGroupSizePolicy.Module(
                                    shape: shape, kind: module is Embedding ? .embedding : .linear)
                            )
                        })
                    let selected = Set(
                        modules.keys.filter {
                            useG128 && ($0.hasSuffix(".down_proj") || $0.hasSuffix(".o_proj"))
                        })
                    let policy = try GemmaGroupSizePolicy(
                        sourceConfiguration: config, modules: modules, selectedModules: selected,
                        omittedTiedHead: true)
                    var options = ModelConversionOptions(
                        bits: 4, groupSize: 64, mode: .affine, calibration: calibration,
                        quantizationPredicate: { path, _ in
                            if selected.contains(path) {
                                return .quantize(
                                    .init(bits: 4, groupSize: 128, mode: .affine, calibration: calibration))
                            }
                            return .quantize()
                        })
                    options.boundedMemory = true
                    let destination = directory.appendingPathComponent("\(calibration)-\(useG128)")
                    let result = try MLXLMCommon.convert(
                        modelDirectory: source, model: model, to: destination, options: options)
                    let exportedConfig = try Data(contentsOf: destination.appendingPathComponent("config.json"))
                    try policy.validateOutputConfiguration(exportedConfig)
                    try policy.validateOutputWeights(result.weightsURLs)
                    var exported = [String: MLXArray]()
                    for shard in result.weightsURLs { exported.merge(try loadArrays(url: shard)) { _, new in new } }
                    XCTAssertFalse(exported.keys.contains { $0.hasPrefix("lm_head.") })
                    for (name, tensor) in arrays where name.contains("norm") {
                        let output = try XCTUnwrap(exported[name])
                        XCTAssertTrue(MLX.all(output .== tensor).item(Bool.self), name)
                    }
                    if useG128 {
                        for (name, tensor) in control {
                            let module = name.split(separator: ".").dropLast().joined(separator: ".")
                            if !selected.contains(module) {
                                let output = try XCTUnwrap(exported[name])
                                XCTAssertEqual(output.shape, tensor.shape, name)
                                XCTAssertTrue(MLX.all(output .== tensor).item(Bool.self), name)
                            }
                        }
                    } else {
                        control = exported
                    }
                    let restored = Gemma3TextModel(configuration)
                    let base = try JSONDecoder().decode(BaseConfiguration.self, from: exportedConfig)
                    try loadWeights(
                        modelDirectory: destination, model: restored, perLayerQuantization: base.perLayerQuantization)
                    let restoredWeights = Dictionary(uniqueKeysWithValues: restored.parameters().flattened())
                    for suffix in ["weight", "scales", "biases"] {
                        let head = try XCTUnwrap(restoredWeights["lm_head." + suffix])
                        let embedding = try XCTUnwrap(restoredWeights["model.embed_tokens." + suffix])
                        XCTAssertTrue(MLX.all(head .== embedding).item(Bool.self), suffix)
                    }
                    let logits = restored(MLXArray([Int32(1), 2]).reshaped(1, 2), cache: nil)
                    XCTAssertEqual(logits.shape, [1, 2, 32])
                    XCTAssertTrue(MLX.all(MLX.isFinite(logits)).item(Bool.self))
                }
            }
        }
    }
}

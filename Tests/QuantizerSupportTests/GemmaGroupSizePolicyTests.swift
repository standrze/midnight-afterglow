import Foundation
import QuantizerSupport
import XCTest

final class GemmaGroupSizePolicyTests: XCTestCase {
    private typealias Policy = GemmaGroupSizePolicy
    private let down = "model.layers.0.mlp.down_proj"
    private let attention = "model.layers.0.self_attn.o_proj"
    private let embedding = "model.embed_tokens"
    private let router = "model.layers.0.router.proj"

    func testQ8TiedHeadRequiresMatchingMetadataAlias() throws {
        var tiedModules = modules()
        tiedModules.removeValue(forKey: "lm_head")
        let policy = try Policy(
            sourceConfiguration: source(), modules: tiedModules, selectedModules: [],
            q8Modules: [embedding], omittedTiedHead: true)
        let original = output(selected: [], q8: [embedding])
        XCTAssertTrue(policy.requiresTiedHeadQuantizationAlias)
        XCTAssertThrowsError(try policy.validateOutputConfiguration(original))
        let finalized = try policy.finalizedOutputConfiguration(original)
        XCTAssertNoThrow(try policy.validateOutputConfiguration(finalized))
        XCTAssertEqual(try policy.finalizedOutputConfiguration(finalized), finalized)
        var configuration = try XCTUnwrap(JSONSerialization.jsonObject(with: finalized) as? [String: Any])
        var quantization = try XCTUnwrap(configuration["quantization"] as? [String: Any])
        let head = try XCTUnwrap(quantization["lm_head"] as? [String: Any])
        let embed = try XCTUnwrap(quantization[embedding] as? [String: Any])
        XCTAssertTrue(NSDictionary(dictionary: head).isEqual(to: embed))
        XCTAssertEqual(head["bits"] as? Int, 8)
        for invalid in [
            ["bits": 4, "group_size": 64, "mode": "affine"],
            ["bits": 8, "group_size": 128, "mode": "affine"],
        ] {
            quantization["lm_head"] = invalid
            configuration["quantization"] = quantization
            configuration["quantization_config"] = quantization
            let data = try JSONSerialization.data(withJSONObject: configuration)
            XCTAssertThrowsError(try policy.validateOutputConfiguration(data))
            XCTAssertThrowsError(try policy.finalizedOutputConfiguration(data))
        }
    }

    func testQ4AndUntiedFinalizationPreservesConfigurationBytes() throws {
        var tiedModules = modules()
        tiedModules.removeValue(forKey: "lm_head")
        let tied = try Policy(
            sourceConfiguration: source(), modules: tiedModules, selectedModules: [], omittedTiedHead: true)
        let q4 = output(selected: [])
        XCTAssertFalse(tied.requiresTiedHeadQuantizationAlias)
        XCTAssertEqual(try tied.finalizedOutputConfiguration(q4), q4)
        let untied = try makePolicy(selected: [], q8: [embedding])
        let q8 = output(selected: [], q8: [embedding])
        XCTAssertFalse(untied.requiresTiedHeadQuantizationAlias)
        XCTAssertEqual(try untied.finalizedOutputConfiguration(q8), q8)
    }

    func testSelectiveGeometryAndAliasesPreserveGemma270MConfiguration() throws {
        let policy = try makePolicy(selected: [down, attention])
        XCTAssertEqual(policy.selectedModules, [down, attention])
        XCTAssertNoThrow(try policy.validateOutputConfiguration(output(selected: [down, attention])))
        XCTAssertNoThrow(try makePolicy(selected: []).validateOutputConfiguration(output(selected: [])))
        for broken in [
            output(selected: [down]),
            output(selected: [down, attention], differingAlias: true),
            output(selected: [down, attention], bits: 3),
            output(selected: [down, attention], mutateSource: true),
        ] {
            XCTAssertThrowsError(try policy.validateOutputConfiguration(broken))
        }
    }

    func testRejectsProtectedUnknownAndConflictingSelections() throws {
        for selected in [[embedding], [router], ["lm_head"], ["missing"]] {
            XCTAssertThrowsError(try makePolicy(selected: Set(selected)))
        }
        XCTAssertThrowsError(try makePolicy(selected: [down], q8: [down]))
        XCTAssertThrowsError(try makePolicy(selected: [down], skipped: [down]))
        let q8Policy = try makePolicy(selected: [down], q8: [router])
        XCTAssertNoThrow(try q8Policy.validateOutputConfiguration(output(selected: [down], q8: [router])))
        XCTAssertThrowsError(try q8Policy.validateOutputConfiguration(output(selected: [down])))
    }

    func testRejectsA4BExpertDownWidthAndInvalidGeometry() throws {
        for width in [0, 64, 704, 2112] {
            var changed = modules()
            changed[down] = .init(shape: [128, 2816, width], kind: .switchLinear)
            XCTAssertThrowsError(
                try Policy(
                    sourceConfiguration: source(), modules: changed, selectedModules: [down]))
        }
        var changed = modules()
        changed[down] = .init(shape: [128, 2816, 2048], kind: .switchLinear)
        XCTAssertNoThrow(try Policy(sourceConfiguration: source(), modules: changed, selectedModules: [down]))
    }

    func testRejectsPrequantizedForeignAndContradictorySourceMetadata() throws {
        for metadata in [
            #""quantization":{"bits":4,"group_size":64},"#,
            #""quantization_config":{"bits":8,"group_size":64},"#,
            #""quantization":null,"#,
            #""compression_config":{"format":"pack-quantized"},"#,
            #""quantization":{"bits":4},"quantization_config":{"bits":3},"#,
        ] {
            let data = Data("{\(metadata)\"model_type\":\"gemma3_text\"}".utf8)
            XCTAssertThrowsError(try Policy(sourceConfiguration: data, modules: modules(), selectedModules: [down]))
        }
        let nested = Data(#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text","quantization":{}}}"#.utf8)
        XCTAssertThrowsError(try Policy(sourceConfiguration: nested, modules: modules(), selectedModules: [down]))
        let foreign = Data(#"{"model_type":"llama"}"#.utf8)
        XCTAssertThrowsError(try Policy(sourceConfiguration: foreign, modules: modules(), selectedModules: [down]))
    }

    func testOutputHeadersMustMatchPackedAndMetadataGeometry() throws {
        try withWorkspace { directory in
            let policy = try makePolicy(selected: [down, attention])
            let good = try writePackedFixture(at: directory.appendingPathComponent("good.safetensors"))
            XCTAssertNoThrow(try policy.validateOutputWeights([good]))
            for mutation in ["wrong_group", "wrong_bits", "wrong_dtype", "missing_bias"] {
                let file = try writePackedFixture(
                    at: directory.appendingPathComponent(mutation + ".safetensors"), mutation: mutation)
                XCTAssertThrowsError(try policy.validateOutputWeights([file]), mutation)
            }
            XCTAssertThrowsError(try policy.validateOutputWeights([good, good]))
        }
    }

    func testExplicitIdenticalHeadCanRemainTiedWithoutConfigFlag() throws {
        try withWorkspace { directory in
            let identical = try sourceFixture(directory: directory, name: "identical", head: [1, 2, 3, 4])
            XCTAssertTrue(try Policy.preservesGemma3TiedHead(sourceConfiguration: source(), shards: [identical]))
            let omitted = try sourceFixture(directory: directory, name: "omitted", head: nil)
            XCTAssertTrue(try Policy.preservesGemma3TiedHead(sourceConfiguration: source(), shards: [omitted]))
            let changed = try sourceFixture(directory: directory, name: "changed", head: [1, 2, 3, 5])
            XCTAssertFalse(try Policy.preservesGemma3TiedHead(sourceConfiguration: source(), shards: [changed]))
            XCTAssertThrowsError(
                try Policy.preservesGemma3TiedHead(sourceConfiguration: source(tie: true), shards: [changed]))
            XCTAssertFalse(
                try Policy.preservesGemma3TiedHead(sourceConfiguration: source(tie: false), shards: [identical]))
            XCTAssertThrowsError(
                try Policy.preservesGemma3TiedHead(sourceConfiguration: source(tie: false), shards: [omitted]))
        }
    }

    func testRejectsPackedPayloadEvenWithoutConfigDeclaration() throws {
        try withWorkspace { directory in
            let file = directory.appendingPathComponent("packed.safetensors")
            try writeTensors([("model.embed_tokens.weight", "U32", [2, 2], Data(repeating: 0, count: 16))], to: file)
            XCTAssertThrowsError(try Policy.preservesGemma3TiedHead(sourceConfiguration: source(), shards: [file]))
        }
    }

    func testSkippedModulesRemainUnpackedAndTiedHeadStaysOmitted() throws {
        try withWorkspace { directory in
            var selectedModules = modules()
            selectedModules.removeValue(forKey: "lm_head")
            let policy = try Policy(
                sourceConfiguration: source(), modules: selectedModules, selectedModules: [down],
                skippedModules: [attention], omittedTiedHead: true)
            var config = try XCTUnwrap(JSONSerialization.jsonObject(with: output(selected: [down])) as? [String: Any])
            var quantization = try XCTUnwrap(config["quantization"] as? [String: Any])
            quantization[attention] = false
            config["quantization"] = quantization
            config["quantization_config"] = quantization
            XCTAssertNoThrow(try policy.validateOutputConfiguration(JSONSerialization.data(withJSONObject: config)))
            let file = try writePackedFixture(at: directory.appendingPathComponent("with-head.safetensors"))
            XCTAssertThrowsError(try policy.validateOutputWeights([file]))
        }
    }

    private func modules() -> [String: Policy.Module] {
        [
            down: .init(shape: [8, 2048], kind: .linear),
            attention: .init(shape: [8, 1024], kind: .linear),
            embedding: .init(shape: [16, 640], kind: .embedding),
            router: .init(shape: [8, 640], kind: .linear),
            "lm_head": .init(shape: [16, 640], kind: .linear),
        ]
    }

    private func makePolicy(
        selected: Set<String>, q8: Set<String> = [], skipped: Set<String> = []
    ) throws -> Policy {
        try Policy(
            sourceConfiguration: source(), modules: modules(), selectedModules: selected,
            q8Modules: q8, skippedModules: skipped)
    }

    private func source(tie: Bool? = nil) -> Data {
        let tieField = tie.map { ",\"tie_word_embeddings\":\($0)" } ?? ""
        return Data(
            "{\"model_type\":\"gemma3_text\",\"hidden_size\":640,\"_sliding_window_pattern\":6,\"layer_types\":[\"sliding_attention\",\"full_attention\"]\(tieField)}"
                .utf8)
    }

    private func output(
        selected: Set<String>, q8: Set<String> = [], differingAlias: Bool = false,
        bits: Int = 4, mutateSource: Bool = false
    ) -> Data {
        var configuration = try! JSONSerialization.jsonObject(with: source()) as! [String: Any]
        var quantization: [String: Any] = ["bits": bits, "group_size": 64, "mode": "affine"]
        for name in selected { quantization[name] = ["bits": bits, "group_size": 128, "mode": "affine"] }
        for name in q8 { quantization[name] = ["bits": 8, "group_size": 64, "mode": "affine"] }
        configuration["quantization"] = quantization
        configuration["quantization_config"] = differingAlias ? ["bits": 4] : quantization
        if mutateSource { configuration["_sliding_window_pattern"] = 5 }
        return try! JSONSerialization.data(withJSONObject: configuration)
    }

    private func writePackedFixture(at file: URL, mutation: String = "") throws -> URL {
        var tensors = [(String, String, [Int], Data)]()
        for (name, module) in modules().sorted(by: { $0.key < $1.key }) {
            let selected = [down, attention].contains(name)
            let groupSize = selected && mutation != "wrong_group" ? 128 : 64
            var packed = module.shape
            var grid = module.shape
            packed[packed.count - 1] /= mutation == "wrong_bits" ? 4 : 8
            grid[grid.count - 1] /= groupSize
            tensors.append((name + ".weight", "U32", packed, Data(repeating: 0, count: packed.reduce(4, *))))
            tensors.append((name + ".scales", "BF16", grid, Data(repeating: 0, count: grid.reduce(2, *))))
            if mutation != "missing_bias" {
                tensors.append(
                    (
                        name + ".biases", mutation == "wrong_dtype" ? "F16" : "BF16", grid,
                        Data(repeating: 0, count: grid.reduce(2, *))
                    ))
            }
        }
        try writeTensors(tensors, to: file)
        return file
    }

    private func sourceFixture(directory: URL, name: String, head: [UInt8]?) throws -> URL {
        let file = directory.appendingPathComponent(name + ".safetensors")
        var tensors: [(String, String, [Int], Data)] = [
            ("model.embed_tokens.weight", "BF16", [1, 2], Data([1, 2, 3, 4]))
        ]
        if let head { tensors.append(("lm_head.weight", "BF16", [1, 2], Data(head))) }
        try writeTensors(tensors, to: file)
        return file
    }

    private func writeTensors(_ tensors: [(String, String, [Int], Data)], to file: URL) throws {
        var header = [String: Any]()
        var payload = Data()
        for (name, dtype, shape, bytes) in tensors {
            header[name] = [
                "dtype": dtype, "shape": shape, "data_offsets": [payload.count, payload.count + bytes.count],
            ]
            payload.append(bytes)
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(json.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }
        data.append(json)
        data.append(payload)
        try data.write(to: file)
    }

    private func withWorkspace(_ body: (URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try body(root)
    }
}

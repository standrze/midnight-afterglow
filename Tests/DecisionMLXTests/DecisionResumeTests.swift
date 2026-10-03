import DecisionModels
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import XCTest

@testable import DecisionMLX

private struct ProbeTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let prefix = [1, 2, 3, 4]
        if text.hasSuffix("A") { return prefix + [32] }
        if text.hasSuffix("B") { return prefix + [33] }
        return prefix
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "probe" }
    func convertTokenToId(_ token: String) -> Int? { token == "A" ? 32 : token == "B" ? 33 : nil }
    func convertIdToToken(_ id: Int) -> String? { id == 32 ? "A" : id == 33 ? "B" : nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2, 3, 4] }
}

final class DecisionResumeTests: XCTestCase {
    func testTrainResumeAndAdapterReloadAgree() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "afterglow-resume-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let config = Data(
            #"{"model_type":"qwen3_5_text","hidden_size":16,"num_hidden_layers":2,"intermediate_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"linear_num_value_heads":2,"linear_num_key_heads":1,"linear_key_head_dim":32,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,"vocab_size":64,"full_attention_interval":2,"tie_word_embeddings":true}"#
                .utf8)
        let configuration = try JSONDecoder().decode(Qwen35TextConfiguration.self, from: config)
        func container() throws -> ModelContainer {
            MLXRandom.seed(17)
            let model = Qwen35TextModel(configuration)
            eval(model)
            return ModelContainer(
                context: ModelContext(
                    configuration: .init(directory: root), model: model,
                    processor: StandInUserInputProcessor(), tokenizer: ProbeTokenizer()))
        }
        let base = root.appendingPathComponent("base")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        try config.write(to: base.appendingPathComponent("config.json"))
        let initial = try container()
        try await initial.perform { context in
            try save(
                arrays: Dictionary(uniqueKeysWithValues: context.model.parameters().flattened()),
                url: base.appendingPathComponent("model.safetensors"))
        }
        let baseHashBefore = try DecisionFiles.modelHash(base)
        let train = root.appendingPathComponent("train.jsonl")
        let dev = root.appendingPathComponent("dev.jsonl")
        let row =
            #"{"id":"ID","source_family":"FAMILY","context":"Evidence","schema":{"allowed":{"type":"boolean","description":"Allowed?"}},"labels":{"allowed":true}}"#
        try Data(
            (row.replacingOccurrences(of: "ID", with: "train").replacingOccurrences(of: "FAMILY", with: "train-family")
                + "\n").utf8
        ).write(to: train)
        try Data(
            (row.replacingOccurrences(of: "ID", with: "dev").replacingOccurrences(of: "FAMILY", with: "dev-family")
                + "\n").utf8
        ).write(to: dev)
        let contract = DecisionModelContract(
            model: "test/qwen", revision: String(repeating: "a", count: 40),
            candidateCodes: ["A", "B"], candidateTokenIDs: [32, 33])
        var options = DecisionTrainingOptions()
        options.rank = 2
        options.epochs = 4
        options.batchSize = 1
        options.learningRate = 0.01
        options.checkpointEvery = 1
        let full = root.appendingPathComponent("full")
        let resumed = root.appendingPathComponent("resumed")
        let fullReport = try await DecisionTrainer.train(
            container: initial, model: base, data: train, development: dev,
            contract: contract, options: options, output: full)
        XCTAssertLessThan(fullReport.finalLoss, fullReport.initialLoss)
        _ = try await DecisionTrainer.train(
            container: container(), model: base, data: train, development: dev,
            contract: contract, options: options, output: resumed, stopAfterUpdates: 2)
        let resumedReport = try await DecisionTrainer.train(
            container: container(), model: base, data: train, development: dev,
            contract: contract, options: options, output: resumed, resume: true)
        XCTAssertEqual(fullReport.updates, resumedReport.updates)
        XCTAssertEqual(fullReport.finalLoss, resumedReport.finalLoss, accuracy: 1e-5)
        let first = try loadArrays(url: full.appendingPathComponent("adapter-step-4/adapters.safetensors"))
        let second = try loadArrays(url: resumed.appendingPathComponent("adapter-step-4/adapters.safetensors"))
        XCTAssertEqual(Set(first.keys), Set(second.keys))
        for key in first.keys { XCTAssertLessThanOrEqual(abs(first[key]! - second[key]!).max().item(Float.self), 1e-5) }
        let expected = try await initial.perform { context in
            try DecisionRuntime.score(
                context: context,
                request: .init(
                    model: "test", context: "Evidence",
                    schema: ["allowed": .init(type: "boolean", description: "Allowed?")]), contract: contract)
        }
        let reloaded = try container()
        let adapterURL = full.appendingPathComponent("adapter-step-4")
        let actual = try await reloaded.perform { context in
            let adapter = try DecisionRuntime.loadAdapter(adapterURL)
            try adapter.load(into: context.model)
            try DecisionRuntime.auditAdapter(adapter, model: context.model)
            return try DecisionRuntime.score(
                context: context,
                request: .init(
                    model: "test", context: "Evidence",
                    schema: ["allowed": .init(type: "boolean", description: "Allowed?")]), contract: contract)
        }
        XCTAssertEqual(actual.output, expected.output)
        for (key, probability) in expected.fields["allowed"]!.probabilities {
            XCTAssertEqual(actual.fields["allowed"]!.probabilities[key]!, probability, accuracy: 1e-5)
        }
        XCTAssertEqual(try DecisionFiles.modelHash(base), baseHashBefore)
    }
}

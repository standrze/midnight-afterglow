import CryptoKit
import DecisionModels
import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

/// An independent full-prompt reference scorer. No conversation cache is retained.
public enum DecisionRuntime {
    public static func load(model: URL, adapter: URL? = nil) async throws -> ModelContainer {
        let container = try await LLMModelFactory.shared.loadContainer(
            from: model, using: #huggingFaceTokenizerLoader())
        if let adapter {
            try await container.perform { context in
                let weights = try loadAdapter(adapter)
                try weights.load(into: context.model)
                try auditAdapter(weights, model: context.model)
                context.model.train(false)
                eval(context.model)
            }
        }
        return container
    }

    public static func prepare(
        request: DecisionRequest, contract: DecisionModelContract,
        tokenizer: any MLXLMCommon.Tokenizer
    ) throws -> [(name: String, tokens: [Int], candidates: [Int])] {
        try contract.validate()
        try request.validate(maximumChoices: contract.candidateCodes.count)
        return try request.names.map { name in
            let prompt = try DecisionPrompt.text(request: request, field: name, contract: contract)
            let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
            guard !tokens.isEmpty, tokens.count <= contract.maxLength else {
                throw DecisionError.invalidRequest(
                    "\(name): prompt has \(tokens.count) tokens; limit \(contract.maxLength). Nothing was truncated.")
            }
            let count = request.schema[name]!.values.count
            let candidates = Array(contract.candidateTokenIDs.prefix(count))
            for index in 0..<count {
                let appended = tokenizer.encode(text: prompt + contract.candidateCodes[index], addSpecialTokens: false)
                guard appended == tokens + [candidates[index]] else {
                    throw DecisionError.invalidContract(
                        "Candidate code \(contract.candidateCodes[index]) is not the contracted single token at the answer boundary."
                    )
                }
            }
            return (name, tokens, candidates)
        }
    }

    public static func score(
        context: ModelContext, request: DecisionRequest,
        contract: DecisionModelContract
    ) throws -> DecisionResponse {
        let prepared = try prepare(request: request, contract: contract, tokenizer: context.tokenizer)
        var fields: [String: DecisionFieldResult] = [:]
        var count = 0
        for row in prepared {
            try Task.checkCancellation()
            let cache = try context.model.newCache(parameters: nil)
            let logits = context.model(MLXArray(row.tokens).reshaped(1, row.tokens.count), cache: cache)
            guard logits.ndim == 3, logits.dim(0) == 1, logits.dim(1) > 0,
                row.candidates.allSatisfy({ $0 < logits.dim(2) })
            else {
                throw DecisionError.invalidContract("Model vocabulary or logits shape differs from decision contract.")
            }
            let candidates = take(logits[0, -1, 0...].asType(.float32), MLXArray(row.candidates), axis: 0)
            eval(candidates)
            let values = candidates.asArray(Float.self).map(Double.init)
            fields[row.name] = try DecisionScoring.result(
                logits: values,
                choices: request.schema[row.name]!.values, temperature: contract.temperature,
                rubric: (request.scoreFields ?? []).contains(row.name))
            count += row.tokens.count
        }
        try Task.checkCancellation()
        return DecisionResponse(
            model: request.model, revision: contract.sourceRevision ?? contract.revision,
            temperature: contract.temperature, fields: fields, promptTokens: count,
            artifactSHA256: contract.adapterFingerprint)
    }

    public static func loadAdapter(_ directory: URL) throws -> LoRAContainer {
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("adapter_model.safetensors").path) {
            let data = try Data(contentsOf: directory.appendingPathComponent("adapter_config.json"))
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            guard (object["bias"] as? String ?? "none") == "none",
                (object["use_rslora"] as? Bool ?? false) == false,
                (object["fan_in_fan_out"] as? Bool ?? false) == false,
                (object["rank_pattern"] as? [String: Any] ?? [:]).isEmpty,
                (object["alpha_pattern"] as? [String: Any] ?? [:]).isEmpty,
                object["modules_to_save"] == nil || object["modules_to_save"] is NSNull
            else {
                throw DecisionError.unsupported("PEFT variant requires unsupported scaling or extra trainable modules.")
            }
            let raw = try loadArrays(url: directory.appendingPathComponent("adapter_model.safetensors"))
            guard !raw.isEmpty,
                raw.keys.allSatisfy({ $0.hasSuffix(".lora_A.weight") || $0.hasSuffix(".lora_B.weight") })
            else {
                throw DecisionError.invalidContract("Unrecognized or empty PEFT tensor set.")
            }
            // Midnight and Afterglow load the text backbone inside Qwen35Model.
            let imported = try LoRAContainer.fromPEFT(directory: directory)
            var mapped: [String: MLXArray] = [:]
            for (key, value) in imported.parameters.flattened() {
                let name = key.replacingOccurrences(of: "model.language_model.", with: "language_model.model.")
                guard mapped[name] == nil else {
                    throw DecisionError.invalidContract("Duplicate adapter tensor mapping.")
                }
                mapped[name] = value
            }
            return LoRAContainer(configuration: imported.configuration, parameters: .unflattened(mapped))
        }
        return try LoRAContainer.from(directory: directory)
    }

    public static func auditAdapter(_ adapter: LoRAContainer, model: any LanguageModel) throws {
        let supplied = Dictionary(uniqueKeysWithValues: adapter.parameters.flattened())
        let actual = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().filter {
                $0.0.hasSuffix(".lora_a") || $0.0.hasSuffix(".lora_b")
            })
        guard Set(supplied.keys) == Set(actual.keys), supplied.allSatisfy({ actual[$0.key]?.shape == $0.value.shape })
        else {
            throw DecisionError.invalidContract("Adapter application is incomplete or tensor shapes differ.")
        }
    }

    /// Convert a publisher PEFT adapter without loading or changing base weights.
    public static func convertAdapter(source: URL, destination: URL, contract: DecisionModelContract) throws {
        let adapter = try loadAdapter(source)
        try DecisionFiles.transaction(destination: destination) { staging in
            try save(
                arrays: Dictionary(uniqueKeysWithValues: adapter.parameters.flattened()),
                url: staging.appendingPathComponent("adapters.safetensors"))
            try DecisionFiles.write(adapter.configuration, to: staging.appendingPathComponent("adapter_config.json"))
            var converted = contract
            converted.adapterFingerprint = try DecisionFiles.hash(
                staging.appendingPathComponent("adapters.safetensors"))
            try DecisionFiles.write(converted, to: staging.appendingPathComponent(DecisionModelContract.filename))
            for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE", "README.md"] {
                let original = source.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: original.path) {
                    try FileManager.default.copyItem(at: original, to: staging.appendingPathComponent(name))
                }
            }
        }
    }
}

/// Atomic artifact writes and streaming fingerprints; originals survive failed exports.
public enum DecisionFiles {
    public static func write<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(value).write(to: url, options: .atomic)
    }

    public static func hash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func modelHash(_ directory: URL) throws -> String {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter {
                $0.pathExtension == "safetensors"
                    || ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"].contains(
                        $0.lastPathComponent)
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard files.contains(where: { $0.pathExtension == "safetensors" }) else {
            throw DecisionError.invalidContract("No base weights.")
        }
        var hash = SHA256()
        for file in files {
            hash.update(data: Data((file.lastPathComponent + ":" + (try self.hash(file)) + "\n").utf8))
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    public static func transaction(destination: URL, operation: (URL) throws -> Void) throws {
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw DecisionError.invalidRequest("Destination exists: \(destination.path)")
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let staging = destination.deletingLastPathComponent().appendingPathComponent(".afterglow-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        try operation(staging)
        try FileManager.default.moveItem(at: staging, to: destination)
    }
}

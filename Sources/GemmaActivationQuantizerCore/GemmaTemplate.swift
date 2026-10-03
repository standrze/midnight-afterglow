import Foundation
import MLX
import MistralActivationScaleSearchCore
import QuantizerSupport

struct GemmaTemplateRecipe: Decodable {
    let algorithm: String
    let sourceModel: String
    let modelType: String
    let bits: Int
    let groupSize: Int
    let mode: String
    let q4ScaleSearchModules: [String]
    let q8Modules: [String]

    enum CodingKeys: String, CodingKey {
        case algorithm, bits, mode
        case sourceModel = "source_model"
        case modelType = "model_type"
        case groupSize = "group_size"
        case q4ScaleSearchModules = "q4_scale_search_modules"
        case q8Modules = "q8_modules"
    }
}

struct GemmaTemplate {
    let directory: URL
    let recipe: GemmaTemplateRecipe
    let quantization: [String: Any]
    let identity: [String: String]

    init(directory: URL, source: URL, sourceConfig: Data) throws {
        self.directory = directory
        let provenance = directory.appendingPathComponent("scale-search-quantization.json")
        let sourceObject = try object(sourceConfig)
        let templateObject = try object(
            Data(contentsOf: directory.appendingPathComponent("config.json")))
        var recipeObject = try object(Data(contentsOf: provenance))
        // Early LS2 provenance omitted mode; require explicit affine policy in both aliases.
        if recipeObject["mode"] == nil,
            recipeObject["algorithm"] as? String == "q4r8_affine_scale_search_ls2",
            let policy = templateObject["quantization"] as? [String: Any],
            let alias = templateObject["quantization_config"] as? [String: Any],
            policy["mode"] as? String == "affine", alias["mode"] as? String == "affine"
        {
            recipeObject["mode"] = "affine"
        }
        recipe = try JSONDecoder().decode(
            GemmaTemplateRecipe.self, from: JSONSerialization.data(withJSONObject: recipeObject))
        guard recipe.algorithm == "q4r8_affine_scale_search_ls2", recipe.bits == 4,
            recipe.groupSize == 64, recipe.mode == "affine",
            URL(fileURLWithPath: recipe.sourceModel).standardizedFileURL.resolvingSymlinksInPath()
                == source,
            Set(recipe.q4ScaleSearchModules).count == recipe.q4ScaleSearchModules.count,
            Set(recipe.q4ScaleSearchModules).isDisjoint(with: recipe.q8Modules)
        else {
            throw QuantizerInputError("Template must be Wick LS2 Q4/G64 from the named BF16 source.")
        }
        guard recipe.modelType == sourceObject["model_type"] as? String,
            NSDictionary(dictionary: stripped(templateObject)).isEqual(to: stripped(sourceObject)),
            let policy = templateObject["quantization"] as? [String: Any],
            let alias = templateObject["quantization_config"] as? [String: Any],
            NSDictionary(dictionary: policy).isEqual(to: alias)
        else {
            throw QuantizerInputError("Template/source configuration or quantization aliases differ.")
        }
        quantization = policy
        identity = try Self.capture(directory)
        let sourceIdentity = try GemmaActivationSourceIdentity.capture(source: source)
        let tokenizerNames = [
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
            "chat_template.jinja",
        ]
        guard
            Set(identity.keys.filter { tokenizerNames.contains($0) })
                == Set(sourceIdentity.tokenizerFingerprints.keys)
        else {
            throw QuantizerInputError("Template/source tokenizer asset inventories differ.")
        }
        for (name, hash) in sourceIdentity.tokenizerFingerprints {
            guard identity[name] == hash else {
                throw QuantizerInputError("Template/source tokenizer differs: \(name)")
            }
        }
    }

    func requireQ4G64(_ path: String) throws {
        if let value = quantization[path], !(value is [String: Any]) {
            throw QuantizerInputError("Selected module is excluded by template policy: \(path)")
        }
        let policy = quantization[path] as? [String: Any] ?? quantization
        guard policy["bits"] as? Int == 4, policy["group_size"] as? Int == 64,
            policy["mode"] as? String == "affine", recipe.q4ScaleSearchModules.contains(path),
            !recipe.q8Modules.contains(path)
        else { throw QuantizerInputError("Selected module must be actual LS2 affine Q4/G64: \(path)") }
    }

    func requireUnchanged() throws {
        guard try Self.capture(directory) == identity else {
            throw QuantizerInputError("Template changed during conversion.")
        }
    }

    func copySidecars(to output: URL) throws {
        for name in identity.keys.sorted()
        where name != "weights" && name != "model.safetensors.index.json" {
            try FileManager.default.copyItem(
                at: directory.appendingPathComponent(name).resolvingSymlinksInPath(),
                to: output.appendingPathComponent(name))
        }
    }

    private static func capture(_ directory: URL) throws -> [String: String] {
        let index = try object(
            Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json")))
        guard let map = index["weight_map"] as? [String: String] else {
            throw QuantizerInputError("Invalid template index.")
        }
        let shards = Set(map.values)
        var result = ["weights": try IndexedSafetensorsFingerprint.compute(directory: directory)]
        for entry in try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        {
            let name = entry.lastPathComponent
            if shards.contains(name) { continue }
            guard name != "gemma-awss-quantization.json", entry.pathExtension != "safetensors" else {
                throw QuantizerInputError("Unexpected unindexed weights or previous AWSS report: \(name)")
            }
            let resolved = entry.resolvingSymlinksInPath()
            guard try resolved.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
                throw QuantizerInputError("Template sidecars must be regular files: \(name)")
            }
            result[name] = GemmaActivationProvenance.fingerprint(try Data(contentsOf: resolved))
        }
        return result
    }
}

private func object(_ data: Data) throws -> [String: Any] {
    guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw QuantizerInputError("Expected JSON configuration object.")
    }
    return result
}

private func stripped(_ object: [String: Any]) -> [String: Any] {
    var result = object
    result.removeValue(forKey: "quantization")
    result.removeValue(forKey: "quantization_config")
    if let text = result["text_config"] as? [String: Any] { result["text_config"] = stripped(text) }
    return result
}

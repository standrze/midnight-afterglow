import Foundation

/// Labeled JSONL record.
///
/// Labels are never part of the model request.
public struct DecisionExample: Sendable {
    public let id: String
    public let sourceFamily: String
    public var request: DecisionRequest
    public let labels: [String: DecisionValue]
}

public enum DecisionDataset {
    public static func read(_ url: URL, contract: DecisionModelContract) throws -> [DecisionExample] {
        let data = try Data(contentsOf: url)
        guard data.count <= 256 * 1_048_576 else { throw DecisionError.invalidRequest("Dataset exceeds 256 MiB.") }
        struct Labels: Decodable {
            let id: String
            let sourceFamily: String
            enum CodingKeys: String, CodingKey {
                case id, labels
                case sourceFamily = "source_family"
            }
            let labels: [String: DecisionValue]
        }
        var seen = Set<String>()
        return try data.split(separator: 10).enumerated().map { line, bytes in
            guard bytes.count <= 1_048_576 else { throw DecisionError.invalidRequest("Dataset row exceeds 1 MiB.") }
            let rowData = Data(bytes)
            let metadata = try JSONDecoder().decode(Labels.self, from: rowData)
            guard !metadata.id.isEmpty, !metadata.sourceFamily.isEmpty, seen.insert(metadata.id).inserted else {
                throw DecisionError.invalidRequest("Missing or duplicate example ID/family at line \(line + 1).")
            }
            let request = try DecisionRequest.decode(rowData)
            try request.validate(maximumChoices: contract.candidateCodes.count)
            guard Set(metadata.labels.keys) == Set(request.schema.keys) else {
                throw DecisionError.invalidRequest("\(metadata.id): labels must cover every field once.")
            }
            for (name, label) in metadata.labels {
                let values = request.schema[name]!.values
                guard
                    values.contains(label)
                        || ((request.scoreFields ?? []).contains(name) && values.contains(.string(label.key)))
                else {
                    throw DecisionError.invalidRequest("\(metadata.id): label for \(name) is outside allowed choices.")
                }
            }
            return DecisionExample(
                id: metadata.id, sourceFamily: metadata.sourceFamily, request: request, labels: metadata.labels)
        }
    }

    public static func validateSplits(_ splits: [[DecisionExample]]) throws {
        guard splits.allSatisfy({ !$0.isEmpty }) else {
            throw DecisionError.invalidRequest("Each dataset split must be nonempty.")
        }
        for index in splits.indices {
            for other in splits.indices where other > index {
                guard Set(splits[index].map(\.id)).isDisjoint(with: Set(splits[other].map(\.id))),
                    Set(splits[index].map(\.sourceFamily)).isDisjoint(with: Set(splits[other].map(\.sourceFamily)))
                else {
                    throw DecisionError.invalidRequest("Example IDs or source families overlap across splits.")
                }
            }
        }
    }
}

/// Evaluation over candidate distributions, independent of the device runtime.
public struct DecisionMetrics: Codable, Sendable {
    public var count = 0
    public var correct = 0
    public var nll = 0.0
    public var brier = 0.0
    public var accuracy: Double { count > 0 ? Double(correct) / Double(count) : 0 }

    public init() {}

    public mutating func record(_ result: DecisionFieldResult, gold: DecisionValue) throws {
        guard let probability = result.probabilities[gold.key] else {
            throw DecisionError.invalidRequest("Gold label missing from prediction.")
        }
        count += 1
        correct += result.value.key == gold.key ? 1 : 0
        nll += -log(max(probability, Double.leastNormalMagnitude))
        brier += result.probabilities.reduce(0) { $0 + pow($1.value - ($1.key == gold.key ? 1 : 0), 2) }
    }
}

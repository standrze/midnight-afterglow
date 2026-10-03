import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct Validate: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Validate labeled datasets and family separation without loading weights.")
    @Option var contract: String
    @Argument var datasets: [String]
    mutating func run() throws {
        guard !datasets.isEmpty else { throw ValidationError("Supply at least one JSONL dataset.") }
        let contract = try JSONDecoder().decode(
            DecisionModelContract.self, from: Data(contentsOf: URL(fileURLWithPath: contract)))
        try contract.validate()
        let splits = try datasets.map { try DecisionDataset.read(URL(fileURLWithPath: $0), contract: contract) }
        try DecisionDataset.validateSplits(splits)
        print("Validated \(splits.map(\.count)) examples; no ID or source-family overlap.")
    }
}

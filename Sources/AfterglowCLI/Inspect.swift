import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct Inspect: ParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Inspect a decision contract and local checkpoint.")
    @OptionGroup var options: ModelOptions
    mutating func run() throws {
        let contract = try options.readContract()
        print("Base: \(contract.model) @ \(contract.revision)")
        print(
            "Task: \(contract.task); choices: \(contract.candidateCodes.count); context: \(contract.maxLength); T=\(contract.temperature)"
        )
        print("Local fingerprint: \(try DecisionFiles.modelHash(options.modelURL))")
    }
}

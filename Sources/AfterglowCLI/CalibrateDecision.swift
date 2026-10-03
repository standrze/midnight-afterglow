import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct CalibrateDecision: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decision", abstract: "Fit temperature on a dedicated labeled calibration split.")
    @OptionGroup var options: ModelOptions
    @Option var data: String
    @Option(help: "Train/development/test splits to check for calibration overlap.") var exclude: [String] = []
    @Option var output: String
    mutating func run() async throws {
        guard !exclude.isEmpty else {
            throw ValidationError("Supply --exclude splits to verify calibration separation.")
        }
        var contract = try options.readContract()
        let examples = try DecisionDataset.read(URL(fileURLWithPath: data), contract: contract)
        let other = try exclude.map { try DecisionDataset.read(URL(fileURLWithPath: $0), contract: contract) }
        try DecisionDataset.validateSplits([examples] + other)
        let container = try await DecisionRuntime.load(model: options.modelURL, adapter: options.adapterURL)
        var records: [(logits: [Double], gold: Int)] = []
        for example in examples {
            let fixed = contract
            let result = try await container.perform { context in
                try DecisionRuntime.score(context: context, request: example.request, contract: fixed)
            }
            for name in example.request.names {
                let values = example.request.schema[name]!.values
                records.append(
                    (
                        values.map { result.fields[name]!.logits[$0.key]! },
                        values.firstIndex { $0.key == example.labels[name]!.key }!
                    ))
            }
        }
        func objective(_ logTemperature: Double) -> Double {
            let temperature = exp(logTemperature)
            return records.reduce(0) { sum, record in
                let maximum = record.logits.max()!
                let denominator = record.logits.reduce(0) { $0 + exp(($1 - maximum) / temperature) }
                return sum + log(denominator) - (record.logits[record.gold] - maximum) / temperature
            } / Double(records.count)
        }
        var low = log(0.05)
        var high = log(20.0)
        for _ in 0..<80 {
            let left = low + (high - low) / 3
            let right = high - (high - low) / 3
            if objective(left) < objective(right) { high = right } else { low = left }
        }
        contract.temperature = exp((low + high) / 2)
        try DecisionFiles.write(contract, to: URL(fileURLWithPath: output))
        print("Fitted T=\(contract.temperature); calibration NLL \(objective(log(contract.temperature))).")
    }
}

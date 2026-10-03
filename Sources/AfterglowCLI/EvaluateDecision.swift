import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct EvaluateDecision: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decision", abstract: "Score labeled decision data and report accuracy, NLL and Brier score.")
    @OptionGroup var options: ModelOptions
    @Option var data: String
    @Option var output: String
    mutating func run() async throws {
        let contract = try options.readContract()
        let examples = try DecisionDataset.read(URL(fileURLWithPath: data), contract: contract)
        try DecisionDataset.validateSplits([examples])
        let container = try await DecisionRuntime.load(model: options.modelURL, adapter: options.adapterURL)
        var metrics = DecisionMetrics()
        var predictions: [DecisionResponse] = []
        for example in examples {
            let result = try await container.perform { context in
                try DecisionRuntime.score(context: context, request: example.request, contract: contract)
            }
            for (name, field) in result.fields { try metrics.record(field, gold: example.labels[name]!) }
            predictions.append(result)
        }
        let count = Double(metrics.count)
        let report = EvaluationReport(
            count: metrics.count, accuracy: metrics.accuracy, nll: metrics.nll / count,
            brier: metrics.brier / count, temperature: contract.temperature, predictions: predictions)
        try DecisionFiles.write(report, to: URL(fileURLWithPath: output))
        print("Accuracy \(report.accuracy), NLL \(report.nll), Brier \(report.brier)")
    }
}

import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct TrainDecision: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decision", abstract: "Native LoRA training over candidate logits.")
    @OptionGroup var options: ModelOptions
    @Option var data: String
    @Option var development: String
    @Option var output: String
    @Option var rank = 16
    @Option var learningRate: Float = 5e-5
    @Option var batchSize = 8
    @Option var epochs = 1
    @Option var seed: UInt64 = 17
    @Option var maxLength = 2048
    @Option var checkpointEvery = 10
    @Option(help: "Stop at this total update count to exercise resume.") var stopAfterUpdates: Int?
    @Flag var resume = false
    mutating func run() async throws {
        var training = DecisionTrainingOptions()
        training.rank = rank
        training.learningRate = learningRate
        training.batchSize = batchSize
        training.epochs = epochs
        training.seed = seed
        training.maxLength = maxLength
        training.checkpointEvery = checkpointEvery
        training.initializationAdapter = options.adapter
        let container = try await DecisionRuntime.load(model: options.modelURL)
        let report = try await DecisionTrainer.train(
            container: container, model: options.modelURL,
            data: URL(fileURLWithPath: data), development: URL(fileURLWithPath: development),
            contract: options.readContract(), options: training, output: URL(fileURLWithPath: output),
            resume: resume, stopAfterUpdates: stopAfterUpdates)
        print("Completed \(report.updates) updates; development loss \(report.initialLoss) → \(report.finalLoss).")
    }
}

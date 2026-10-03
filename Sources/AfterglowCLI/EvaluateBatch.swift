import ArgumentParser
import Foundation
import ModelEvaluation

#if canImport(Darwin)
    import Darwin
#endif

struct EvaluateBatch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "batch", abstract: "Run a pinned serial model suite with owned native servers.")
    @Option(help: "Pinned model, runtime, suite and predecessor plan JSON.") var plan: String
    @Option(help: "Batch output root; existing progress is protected.") var output: String

    mutating func run() async throws {
        let planURL = URL(fileURLWithPath: plan)
        let directory = URL(fileURLWithPath: output)
        let worker = Task { try await EvaluationBatch.run(planURL: planURL, output: directory) }
        let priorInterrupt = signal(SIGINT, SIG_IGN)
        let priorTerminate = signal(SIGTERM, SIG_IGN)
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        let terminate = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
        interrupt.setEventHandler { worker.cancel() }
        terminate.setEventHandler { worker.cancel() }
        interrupt.resume()
        terminate.resume()
        defer {
            interrupt.cancel()
            terminate.cancel()
            signal(SIGINT, priorInterrupt)
            signal(SIGTERM, priorTerminate)
        }
        try await worker.value
    }
}

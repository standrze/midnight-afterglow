import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct Evaluate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [
        EvaluateDecision.self, EvaluateModel.self, EvaluateBatch.self, EvaluateRuntime.self,
        EvaluateRuntimeDiagnostic.self,
    ])
}

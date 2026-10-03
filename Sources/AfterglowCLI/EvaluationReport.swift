import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct EvaluationReport: Codable {
    var count: Int
    var accuracy: Double
    var nll: Double
    var brier: Double
    var temperature: Double
    var predictions: [DecisionResponse]
}

import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct Train: AsyncParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [TrainDecision.self])
}

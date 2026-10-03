import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct Calibrate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(subcommands: [CalibrateDecision.self])
}

import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

@main
struct Afterglow: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "midnight-afterglow", abstract: "Train, calibrate and prepare models for Midnight.",
        version: "0.1.0",
        subcommands: [
            Inspect.self, Validate.self, Train.self, Evaluate.self, Calibrate.self,
            Convert.self, Export.self, ModelQuantizer.self, QuantizationFormats.self, Interface.self,
        ], defaultSubcommand: Interface.self)
}

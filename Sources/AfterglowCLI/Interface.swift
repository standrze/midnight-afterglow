import AfterglowConsole
import ArgumentParser
import DecisionMLX
import DecisionModels
import Foundation
import QuantizationCommands

struct Interface: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ui", abstract: "Simple loom/weft training and quantization interface.")
    mutating func run() async throws {
        guard AfterglowTerminal.isInteractive else {
            print(Afterglow.helpMessage())
            return
        }
        try await AfterglowTerminal().run()
    }
}

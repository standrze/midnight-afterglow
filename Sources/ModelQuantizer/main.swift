import ArgumentParser
import QuantizationCommands

@main
struct QuantizationCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "wick", subcommands: [ModelQuantizer.self, QuantizationFormats.self],
        defaultSubcommand: ModelQuantizer.self)
}

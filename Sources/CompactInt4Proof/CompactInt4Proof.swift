import CompactInt4
import Foundation

@main
struct CompactInt4Proof {
    private struct MatrixFixture: Decodable {
        let rows: Int
        let columns: Int
        let weights: [Float]
        let source: String
        let importance: [Float]?
    }

    static func main() {
        do { try run() } catch {
            FileHandle.standardError.write(Data("afterglow-int4-proof: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() throws {
        guard (2...5).contains(CommandLine.arguments.count) else {
            throw CompactInt4Error.invalid(
                "Usage: afterglow-int4-proof OUTPUT.agq4 [cpu|compile-metal|metal] [MATRIX.json] [e8m0|e4m4] (must not exist)"
            )
        }
        let verifying = CommandLine.arguments[1] == "verify"
        guard !verifying || CommandLine.arguments.count >= 3 else {
            throw CompactInt4Error.invalid(
                "Usage: afterglow-int4-proof verify INPUT.agq4 [cpu|compile-metal|metal] [MATRIX.json]")
        }
        let modeIndex = verifying ? 3 : 2
        let fixtureIndex = verifying ? 4 : 3
        let mode = CommandLine.arguments.count > modeIndex ? CommandLine.arguments[modeIndex] : "cpu"
        guard ["cpu", "compile-metal", "metal"].contains(mode) else {
            throw CompactInt4Error.invalid("Unknown execution mode")
        }
        var metalErrors: [String: Double] = [:]
        let fixture =
            CommandLine.arguments.count > fixtureIndex
            ? try JSONDecoder().decode(
                MatrixFixture.self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[fixtureIndex])))
            : nil
        let matrix: CompactInt4Matrix
        if verifying {
            matrix = try CompactInt4Matrix.readArtifact(
                Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])))
            if let fixture {
                guard fixture.rows == matrix.rows, fixture.columns == matrix.columns,
                    fixture.weights.count == matrix.rows * matrix.columns, fixture.weights.allSatisfy(\.isFinite)
                else { throw CompactInt4Error.invalid("Fixture does not match artifact geometry") }
            }
        } else {
            let rows = fixture?.rows ?? 16
            let columns = fixture?.columns ?? 256
            let weights =
                fixture?.weights
                ?? (0..<4096).map { sin(Float($0) * 0.123) * 0.2 + cos(Float($0) * 0.019) * 0.03 }
            guard
                let encoding = CompactInt4Matrix.ScaleEncoding(
                    rawValue: CommandLine.arguments.count == 5 ? CommandLine.arguments[4] : "e8m0")
            else { throw CompactInt4Error.invalid("Unknown scale encoding") }
            matrix = try CompactInt4Matrix.fit(
                weights, rows: rows, columns: columns, importance: fixture?.importance, scaleEncoding: encoding)
            try matrix.writeArtifact(to: URL(fileURLWithPath: CommandLine.arguments[1]))
        }
        let loaded = try CompactInt4Matrix.readArtifact(matrix.artifactData())
        let rows = loaded.rows
        let columns = loaded.columns
        let encoding = loaded.scaleEncoding
        let weights =
            fixture?.weights
            ?? (verifying
                ? loaded.reconstructed()
                : (0..<4096).map { sin(Float($0) * 0.123) * 0.2 + cos(Float($0) * 0.019) * 0.03 })
        let input = (0..<(4 * columns)).map { sin(Float($0) * 0.731) }
        let output = try loaded.multiply(input, batch: 4)
        let dense = loaded.reconstructed()
        var maxError: Double = 0
        for sample in 0..<4 {
            for row in 0..<rows {
                let oracle = (0..<columns).reduce(0.0) {
                    $0 + Double(input[sample * columns + $1]) * Double(dense[row * columns + $1])
                }
                maxError = max(maxError, abs(Double(output[sample * rows + row]) - oracle))
            }
        }
        guard maxError < 1e-4 else { throw CompactInt4Error.invalid("Packed execution failed dense oracle") }
        let squaredError = zip(weights, dense).reduce(0.0) { $0 + pow(Double($1.0 - $1.1), 2) }
        let energy = weights.reduce(0.0) { $0 + pow(Double($1), 2) }
        #if canImport(Metal)
            if mode != "cpu" {
                let backends: [CompactInt4Metal.Backend] =
                    loaded.codeEncoding == .normal16 ? [.packed] : [.packed, .nativeInt4]
                for backend in backends {
                    let runtime = try CompactInt4Metal(
                        backend: backend, scaleEncoding: encoding, codeEncoding: loaded.codeEncoding)
                    if mode == "metal" {
                        let halfInput = input.map(Float16.init)
                        let reference = try loaded.multiply(halfInput.map(Float.init), batch: 4)
                        let actual = try runtime.multiply(loaded, input: halfInput, batch: 4)
                        let error = zip(actual, reference).map { abs(Double($0 - $1)) }.max() ?? 0
                        guard error < 1e-4 else {
                            throw CompactInt4Error.invalid("Metal oracle failed: \(backend): \(error)")
                        }
                        metalErrors[backend.rawValue] = error
                    }
                }
            }
        #else
            guard mode == "cpu" else { throw CompactInt4Error.invalid("Metal unavailable on this platform") }
        #endif
        let report: [String: Any] = [
            "metal_max_abs_errors": metalErrors, "execution_mode": mode,
            "format": matrix.artifactFormat, "scale_encoding": encoding.rawValue,
            "code_encoding": loaded.codeEncoding.rawValue,
            "unsupported_backends": loaded.codeEncoding == .normal16 ? ["nativeInt4"] : [],
            "rows": matrix.rows, "columns": matrix.columns,
            "payload_bytes": matrix.payloadBytes, "bits_per_weight": matrix.bitsPerWeight,
            "artifact_bytes": try matrix.artifactData().count,
            "weight_relative_squared_error": verifying && fixture == nil ? NSNull() : squaredError / energy,
            "packed_cpu_max_abs_error": maxError,
            "source": fixture?.source ?? (verifying ? "input artifact" : "deterministic synthetic fixture"),
            "model_quality_measured": false, "speedup_measured": false,
        ]
        let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: json, as: UTF8.self))
    }
}

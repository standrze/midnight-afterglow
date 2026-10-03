import CompactInt4
import Foundation
import MLX
import Metal

// Research-only payload. Normal16 has no published/library artifact format.
private struct ScreenWeights: Decodable {
    let rows: Int
    let columns: Int
    let codes: [UInt8]
    let scaleBytes: [UInt8]
    let offsets: [UInt16]
    let gains: [UInt16]
    let codebookBits: [UInt16]?
    let polynomialCoefficients: [Float]?

    init(_ matrix: CompactInt4Matrix) {
        rows = matrix.rows
        columns = matrix.columns
        codes = matrix.codes
        scaleBytes = matrix.scaleBytes
        offsets = matrix.offsets
        gains = matrix.gains
        codebookBits = matrix.codeEncoding == .normal16 ? CompactInt4Matrix.normal16Bits : nil
        polynomialCoefficients = nil
    }

    init(repeating small: Self, count: Int) {
        rows = small.rows * count
        columns = small.columns
        codes = Array(repeating: small.codes, count: count).flatMap { $0 }
        scaleBytes = Array(repeating: small.scaleBytes, count: count).flatMap { $0 }
        offsets = Array(repeating: small.offsets, count: count).flatMap { $0 }
        gains = Array(repeating: small.gains, count: count).flatMap { $0 }
        codebookBits = small.codebookBits
        polynomialCoefficients = small.polynomialCoefficients
    }

    func validate() throws {
        let (count, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard rows > 0, columns > 0, columns % 64 == 0, !overflow,
            codes.count == count / 2, scaleBytes.count == count / 32,
            offsets.count == count / 64, gains.count == rows,
            codebookBits == nil || codebookBits!.count == 16,
            offsets.allSatisfy({ Self.bf16($0).isFinite }),
            gains.allSatisfy({ Self.bf16($0).isFinite && Self.bf16($0) > 0 }),
            reconstructed().allSatisfy(\.isFinite)
        else { throw CompactInt4Error.invalid("Invalid research screen payload") }
    }

    static func bf16(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }

    func reconstructed() -> [Float] {
        (0..<(rows * columns)).map { index in
            let nibble = Int((codes[index / 2] >> ((index % 2) * 4)) & 15)
            let code: Float
            let byte = scaleBytes[index / 32]
            let scale: Float
            if let codebookBits {
                code = Float(Float16(bitPattern: codebookBits[nibble]))
                scale = Float(bitPattern: (UInt32(byte >> 4) + 120) << 23 | UInt32(byte & 15) << 19)
            } else {
                code = Float(nibble >= 8 ? nibble - 16 : nibble)
                scale = Float(
                    bitPattern: UInt32(byte & 128) << 24 | (UInt32((byte >> 4) & 7) + 124) << 23 | UInt32(
                        byte & 15) << 19)
            }
            return (code * scale + Self.bf16(offsets[index / 64])) * Self.bf16(gains[index / columns])
        }
    }

    func multiply(_ input: [Float], batch: Int) -> [Float] {
        let weights = reconstructed()
        return (0..<(batch * rows)).map { index in
            let sample = index / rows
            let row = index % rows
            return Float(
                (0..<columns).reduce(0.0) {
                    $0 + Double(input[sample * columns + $1]) * Double(weights[row * columns + $1])
                })
        }
    }
}

@main
struct RuntimeScreen {
    static func main() throws {
        try Device.withDefaultDevice(.gpu) { try run() }
    }

    static func run() throws {
        guard CommandLine.arguments.count >= 3 else {
            throw CompactInt4Error.invalid("Usage: runtime-screen FIXTURE_DIR SHADER_PATH [OPTIONS]")
        }
        let normal16 = CommandLine.arguments.contains("--normal16")
        guard !normal16 || CommandLine.arguments.contains("--packed-nax-options-only") else {
            throw CompactInt4Error.invalid("Normal16 is supported only by the packed NAX research screen")
        }
        let folder = URL(fileURLWithPath: CommandLine.arguments[1])
        let source = try String(contentsOfFile: CommandLine.arguments[2], encoding: .utf8)
        let small: ScreenWeights
        if normal16 {
            let artifact = folder.appendingPathComponent("weights.agq4")
            if FileManager.default.fileExists(atPath: artifact.path) {
                let decoded = try CompactInt4Matrix.readArtifact(Data(contentsOf: artifact))
                guard decoded.codeEncoding == .normal16 else { throw CompactInt4Error.invalid("Expected AGQ4N001") }
                small = ScreenWeights(decoded)
            } else {
                small = try JSONDecoder().decode(
                    ScreenWeights.self, from: Data(contentsOf: folder.appendingPathComponent("normal16.json")))
            }
            guard small.codebookBits != nil else { throw CompactInt4Error.invalid("Missing codebook") }
        } else {
            small = ScreenWeights(
                try CompactInt4Matrix.readArtifact(
                    Data(contentsOf: folder.appendingPathComponent("weights.agq4"))))
        }
        try small.validate()
        let n: Int
        if let index = CommandLine.arguments.firstIndex(of: "--rows") {
            guard index + 1 < CommandLine.arguments.count,
                let count = Int(CommandLine.arguments[index + 1]),
                count > 0, count <= 16_384
            else { fatalError("--rows must be within 1...16384") }
            n = count
        } else {
            n = 2048
        }
        let k = small.columns
        let repetitions = n / small.rows
        guard n % small.rows == 0, let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue()
        else { fatalError("Unsupported screen") }
        let matrix = ScreenWeights(repeating: small, count: repetitions)
        let bytes = [UInt8](try Data(contentsOf: folder.appendingPathComponent("input-f16.bin")))
        let input: [Float16] = stride(from: 0, to: bytes.count, by: 2).map {
            Float16(bitPattern: UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8)
        }
        guard input.count == 128 * k else { fatalError("Unexpected input") }
        func buffer<T>(_ values: [T]) -> any MTLBuffer {
            values.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
            }
        }
        let buffers = [
            buffer(input), buffer(matrix.codes), buffer(matrix.scaleBytes), buffer(matrix.offsets),
            buffer(matrix.gains), buffer([Float](repeating: .nan, count: 128 * n + 64)),
        ]
        let halfOutput = buffer([Float16](repeating: .nan, count: 128 * n + 64))
        var pipelines: [(String, any MTLComputePipelineState, Int, Bool, Int, Int)] = []
        let compileDispatch = CommandLine.arguments.contains("--compile-dispatch")
        let groupOnly = CommandLine.arguments.contains("--mlx-group-options-only")
        let polynomialOnly = groupOnly || CommandLine.arguments.contains("--mlx-polynomial-decode-options-only")
        let lookupOnly = polynomialOnly || CommandLine.arguments.contains("--mlx-normal-lookup-options-only")
        let mlxPrefillOnly = CommandLine.arguments.contains("--mlx-normal-prefill-options-only")
        let mlxDecodeOnly = lookupOnly || CommandLine.arguments.contains("--mlx-normal-decode-options-only")
        let decodeOnly = mlxDecodeOnly || CommandLine.arguments.contains("--normal-decode-options-only")
        guard !compileDispatch || mlxDecodeOnly || mlxPrefillOnly else {
            throw CompactInt4Error.invalid("Compiled dispatch applies only to MLX custom-kernel screens")
        }
        let wideNax = CommandLine.arguments.contains("--nax-wide-options-only")
        let packedNax = wideNax || CommandLine.arguments.contains("--packed-nax-options-only")
        let nax = packedNax || CommandLine.arguments.contains("--nax-options-only")
        let registers = CommandLine.arguments.contains("--register-options-only")
        let relaxed = nax || registers || CommandLine.arguments.contains("--relaxed-options-only")
        let paired = relaxed || CommandLine.arguments.contains("--paired-options-only")
        let newOnly = paired || CommandLine.arguments.contains("--new-kernels-only")
        let cases: [(String, Int, Bool, Bool, Int)]
        if mlxDecodeOnly || mlxPrefillOnly {
            cases = []
        } else if decodeOnly {
            cases = [
                ("normal16_decode_rows2", 6, false, true, 2),
                ("normal16_decode_rows4", 6, false, true, 4),
                ("normal16_decode_rows8", 6, false, true, 8),
            ]
        } else if wideNax {
            cases = [
                ("packed_normal16_bm64_bn64", 5, false, true, 64),
                ("packed_normal16_bm128_bn64", 5, false, true, 128),
                ("packed_normal16_bm64_bn128", 5, false, true, 64),
            ]
        } else if nax {
            cases = [
                (
                    packedNax ? "mlx_nax_packed_loader_bm32_fp16" : "mlx_nax_custom_loader_bm32_fp16", 5,
                    false, true, 32
                ),
                (
                    packedNax ? "mlx_nax_packed_loader_bm64_fp16" : "mlx_nax_custom_loader_bm64_fp16", 5,
                    false, true, 64
                ),
            ]
        } else if registers {
            cases = [
                ("register_dequant_k16_fp16", 4, false, true, 16),
                ("register_dequant_k32_fp16", 4, false, true, 32),
            ]
        } else if relaxed {
            cases = [
                ("native_int4_shared_relaxed_fp16", 1, true, true, 16),
                ("tile_static_relaxed_m16_fp16", 3, false, true, 16),
                ("tile_static_relaxed_m32_fp16", 3, false, true, 32),
            ]
        } else if paired {
            cases = [
                ("simd_rows8_fp16", 2, false, true, 8),
                ("tile_static_m16_fp16", 3, false, true, 16),
                ("tile_static_m32_fp16", 3, false, true, 32),
                ("tile_static_m64_fp16", 3, false, true, 64),
            ]
        } else if newOnly {
            cases = [
                ("simd_decode_fp16", 2, false, true, 1),
                ("tile_dequant_m16_fp16", 3, false, true, 16),
                ("tile_dequant_m32_fp16", 3, false, true, 32),
            ]
        } else {
            cases = [
                ("packed_scalar", 0, false, false, 1), ("native_int4_original", 1, false, false, 16),
                ("native_int4_shared_sums", 1, true, false, 16),
                ("native_int4_shared_sums_fp16", 1, true, true, 16),
            ]
        }
        for (name, kind, cached, half, tileM) in cases {
            let native = kind == 1 || kind >= 3
            let tileN = name.contains("bn128") ? 128 : 64
            let options = MTLCompileOptions()
            options.mathMode = .safe
            options.mathFloatingPointFunctions = .precise
            if native {
                guard #available(macOS 26.4, *), device.supportsFamily(.apple10) else {
                    fatalError("Native INT4 screen requires macOS 26.4 and Apple GPU family 10")
                }
                options.languageVersion = .version4_0
            }
            options.preprocessorMacros = [
                "NATIVE_INT4": NSNumber(value: native ? 1 : 0),
                "SCALE_E3M4": NSNumber(value: normal16 ? 0 : 1),
                "SCALE_E4M4": NSNumber(value: normal16 ? 1 : 0),
                "NORMAL16": NSNumber(value: normal16 ? 1 : 0),
                "CACHE_INPUT_SUMS": NSNumber(value: cached ? 1 : 0),
                "OUTPUT_HALF": NSNumber(value: half ? 1 : 0),
                "TILE_M": NSNumber(value: kind == 6 ? 16 : tileM),
                "REG_K": NSNumber(value: kind == 4 ? tileM : 32),
                "NORMAL_DECODE_ROWS": NSNumber(value: tileM), "NAX_BM": NSNumber(value: kind == 6 ? 64 : tileM),
                "NAX_BN": NSNumber(value: tileN),
                "PACKED_NAX": NSNumber(value: packedNax ? 1 : 0),
                "STATIC_TILE": NSNumber(value: paired ? 1 : 0),
                "RELAXED_PRECISION": NSNumber(value: relaxed ? 1 : 0),
            ]
            let library = try device.makeLibrary(source: source, options: options)
            let functionName =
                kind == 6
                ? "compact_normal16_decode"
                : kind == 5
                    ? "compact_int4_nax"
                    : (kind == 4
                        ? "compact_int4_registers"
                        : (kind == 3
                            ? "compact_int4_tile"
                            : (kind == 2
                                ? (tileM == 8 ? "compact_int4_simd_rows8" : "compact_int4_simd")
                                : (kind == 1 ? "compact_int4_native" : "compact_int4_packed"))))
            let function = library.makeFunction(name: functionName)!
            pipelines.append(
                (name, try device.makeComputePipelineState(function: function), kind, half, tileM, tileN))
        }
        let arrays = try loadArrays(url: folder.appendingPathComponent("native-control.safetensors"))
        let packed = tiled(arrays["packed"]!, repetitions: [repetitions, 1])
        let scales = tiled(arrays["scales"]!, repetitions: [repetitions, 1])
        let biases = tiled(arrays["bias"]!, repetitions: [repetitions, 1])
        eval(packed, scales, biases)
        var results: [[String: Any]] = []
        for batch in (decodeOnly ? [1] : ((wideNax || mlxPrefillOnly) ? [128] : [1, 16, 128])) {
            let originalReference = small.multiply(
                Array(input.prefix(batch * k)).map(Float.init), batch: batch)
            let mxInput = MLXArray(Array(input.prefix(batch * k)), [batch, k])
            eval(mxInput)
            var controlNumericRMSE: Double?
            let compiledControl: ([MLXArray]) -> [MLXArray] = compile { args in
                [
                    quantizedMM(
                        args[0], args[1], scales: args[2], biases: args[3], transpose: true, groupSize: 64, bits: 4)
                ]
            }
            func mlxDispatch(check: Bool = false) -> Double {
                let start = DispatchTime.now().uptimeNanoseconds
                let result =
                    compileDispatch
                    ? compiledControl([mxInput, packed, scales, biases])[0]
                    : quantizedMM(
                        mxInput, packed, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
                eval(result)
                Stream.gpu.synchronize()
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                if check {
                    let values = result.asType(.float32).asArray(Float.self)
                    let reference = arrays["reference"]!.asArray(Float.self)
                    var squared: Double = 0
                    var energy: Double = 0
                    for sample in 0..<batch {
                        for row in 0..<n {
                            let target = reference[sample * small.rows + row % small.rows]
                            let difference = Double(values[sample * n + row]) - Double(target)
                            squared += difference * difference
                            energy += Double(target) * Double(target)
                        }
                    }
                    let rmse = sqrt(squared / energy)
                    guard rmse < 0.003 else { fatalError("Paired control numeric oracle failed") }
                    controlNumericRMSE = rmse
                }
                return elapsed
            }
            if paired { _ = mlxDispatch(check: true) }
            if mlxDecodeOnly || mlxPrefillOnly {
                guard let book = small.codebookBits else { fatalError("Normal16 decode scope") }
                let customInputs: [MLXArray] = [
                    mxInput, MLXArray(matrix.codes), MLXArray(matrix.scaleBytes),
                    MLXArray(matrix.offsets), MLXArray(matrix.gains),
                ]
                eval(customInputs)
                let customReference: [Float]
                if mlxPrefillOnly {
                    let weights = small.reconstructed().map { Float(Float16($0)) }
                    customReference = (0..<(batch * small.rows)).map { index in
                        let sample = index / small.rows
                        let row = index % small.rows
                        return Float(
                            (0..<k).reduce(0.0) {
                                $0 + Double(input[sample * k + $1]) * Double(weights[row * k + $1])
                            })
                    }
                } else {
                    customReference = originalReference
                }
                let customCases: [(Int, Int, Int)]
                if groupOnly {
                    guard small.polynomialCoefficients?.count == 2 else {
                        throw CompactInt4Error.invalid("Missing cubic coefficients")
                    }
                    customCases = [(2, 3, 1), (2, 3, 2), (2, 3, 4)]
                } else if polynomialOnly {
                    guard small.polynomialCoefficients?.count == 2 else {
                        throw CompactInt4Error.invalid("Missing cubic coefficients")
                    }
                    customCases = [(2, 0, 1), (2, 3, 1)]
                } else if lookupOnly {
                    customCases = (compileDispatch ? [(2, 0), (2, 1)] : [(2, 0), (2, 1), (2, 2)]).map {
                        ($0.0, $0.1, 1)
                    }
                } else {
                    customCases = (mlxPrefillOnly ? [64, 128] : [2, 4]).map { ($0, 0, 1) }
                }
                for (rows, lookup, groups) in customCases {
                    let kernel =
                        try mlxPrefillOnly
                        ? Normal16MLXPrefill.kernel(tileM: rows, shader: source)
                        : Normal16MLXDecode.kernel(
                            rows: rows, book: book, lookup: lookup, polynomial: small.polynomialCoefficients,
                            simdGroups: groups)
                    func kernelOutput(_ args: [MLXArray], check: Bool) -> [MLXArray] {
                        return kernel(
                            args,
                            template: mlxPrefillOnly
                                ? [("BM", rows), ("M", batch), ("N", n), ("K", k)]
                                : [("ROWS", rows), ("SG", groups), ("N", n), ("K", k)],
                            grid: mlxPrefillOnly
                                ? (((n + 63) / 64) * 128, (batch + rows - 1) / rows, 1)
                                : (((n + rows * groups - 1) / (rows * groups)) * 32 * groups, 1, 1),
                            threadGroup: (mlxPrefillOnly ? 128 : 32 * groups, 1, 1),
                            outputShapes: [[batch * n + (check ? 64 : 0)]], outputDTypes: [.float16],
                            initValue: check ? .nan : nil)
                    }
                    let compiledWarm: ([MLXArray]) -> [MLXArray] = compile { args in kernelOutput(args, check: true) }
                    let compiledTimed: ([MLXArray]) -> [MLXArray] = compile { args in kernelOutput(args, check: false) }
                    func customDispatch(check: Bool = false) -> (Double, Float) {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let output =
                            compileDispatch
                            ? (check ? compiledWarm(customInputs)[0] : compiledTimed(customInputs)[0])
                            : kernelOutput(customInputs, check: check)[0]
                        eval(output)
                        Stream.gpu.synchronize()
                        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                        var error: Float = 0
                        if check {
                            let values = output.asArray(Float16.self)
                            for row in 0..<(batch * n) {
                                guard values[row].isFinite else { fatalError("MLX decode non-finite") }
                                error = max(
                                    error,
                                    abs(
                                        Float(values[row])
                                            - Float(Float16(customReference[(row / n) * small.rows + row % small.rows]))
                                    )
                                )
                            }
                            guard error < 1e-3, values[(batch * n)...].allSatisfy(\.isNaN) else {
                                fatalError("MLX decode packed oracle failed")
                            }
                        }
                        return (elapsed, error)
                    }
                    let warmup = customDispatch(check: true)
                    var settlingDispatches = 0
                    if CommandLine.arguments.contains("--settle-gpu") {
                        let start = DispatchTime.now().uptimeNanoseconds
                        while DispatchTime.now().uptimeNanoseconds - start < 100_000_000 {
                            _ = mlxDispatch()
                            _ = customDispatch()
                            settlingDispatches += 2
                        }
                    }
                    var custom: [Double] = []
                    var control: [Double] = []
                    for trial in 0..<3 {
                        if trial % 2 == 1 { control.append(mlxDispatch()) }
                        custom.append(customDispatch().0)
                        if trial % 2 == 0 { control.append(mlxDispatch()) }
                    }
                    results.append([
                        "backend": mlxPrefillOnly
                            ? "mlx_custom_normal16_prefill_bm\(rows)"
                            : "mlx_custom_normal16_decode_rows\(rows)_lookup\(lookup)_sg\(groups)",
                        "M": batch, "N": n, "K": k,
                        "resident_wall_seconds": custom, "paired_control_wall_seconds": control,
                        "paired_order": "AB BA AB", "max_abs_error_vs_cpu": Double(warmup.1),
                        "control_numeric_relative_rmse": controlNumericRMSE ?? -1,
                        "output_dtype": "FP16", "weight_tile_rounding": mlxPrefillOnly ? "FP16" : "none",
                        "settling_dispatches": settlingDispatches, "compiled_dispatch": compileDispatch,
                    ])
                }
            }
            for (name, pipeline, kind, half, tileM, tileN) in pipelines {
                let outputTileM = kind == 4 ? 16 : tileM
                if kind == 2 && batch != 1 { continue }
                if relaxed && !decodeOnly && batch == 1 { continue }
                if paired && kind >= 3 && kind != 5 && kind != 6 && batch % outputTileM != 0 { continue }
                func dispatch() throws -> (Double, Double) {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let command = queue.makeCommandBuffer()!
                    let encoder = command.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(pipeline)
                    for (index, buffer) in buffers.enumerated() {
                        encoder.setBuffer(index == 5 && half ? halfOutput : buffer, offset: 0, index: index)
                    }
                    var shape = SIMD4<UInt32>(UInt32(batch), UInt32(n), UInt32(k), 0)
                    encoder.setBytes(&shape, length: MemoryLayout.size(ofValue: shape), index: 6)
                    let threads = MTLSize(
                        width: pipeline.threadExecutionWidth * (kind == 5 ? 4 : 1), height: 1, depth: 1)
                    if kind == 6 {
                        encoder.dispatchThreadgroups(
                            MTLSize(width: (n + tileM - 1) / tileM, height: batch, depth: 1),
                            threadsPerThreadgroup: threads)
                    } else if kind == 1 || kind >= 3 {
                        encoder.dispatchThreadgroups(
                            MTLSize(
                                width: kind == 5 ? (n + tileN - 1) / tileN : (n + 31) / 32,
                                height: (batch + outputTileM - 1) / outputTileM, depth: 1),
                            threadsPerThreadgroup: threads)
                    } else if kind == 2 {
                        encoder.dispatchThreadgroups(
                            MTLSize(width: (n + tileM - 1) / tileM, height: batch, depth: 1),
                            threadsPerThreadgroup: threads)
                    } else {
                        encoder.dispatchThreads(
                            MTLSize(width: batch * n, height: 1, depth: 1), threadsPerThreadgroup: threads)
                    }
                    encoder.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                    guard command.status == .completed else { throw command.error! }
                    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                    return (elapsed, command.gpuEndTime - command.gpuStartTime)
                }
                _ = try dispatch()
                let output = buffers[5].contents().bindMemory(to: Float.self, capacity: 128 * n + 64)
                let halfPointer = halfOutput.contents().bindMemory(to: Float16.self, capacity: 128 * n + 64)
                func value(_ index: Int) -> Float { half ? Float(halfPointer[index]) : output[index] }
                var error: Float = 0
                let reference: [Float]
                if kind >= 3 && kind != 6 {
                    let weight = small.reconstructed().map { Float(Float16($0)) }
                    reference = (0..<(batch * small.rows)).map { index in
                        let sample = index / small.rows
                        let row = index % small.rows
                        return Float(
                            (0..<k).reduce(0.0) {
                                $0 + Double(input[sample * k + $1]) * Double(weight[row * k + $1])
                            })
                    }
                } else {
                    reference = originalReference
                }
                let scale = reference.map { abs($0) }.max() ?? 1
                for sample in 0..<batch {
                    for row in 0..<n {
                        guard value(sample * n + row).isFinite else { fatalError("Non-finite result") }
                        error = max(
                            error,
                            abs(
                                value(sample * n + row)
                                    - (half
                                        ? Float(Float16(reference[sample * small.rows + row % small.rows]))
                                        : reference[sample * small.rows + row % small.rows])))
                    }
                }
                guard error < (half ? max(1e-3, scale * 1e-3) : 1e-3) else {
                    fatalError("Oracle failed \(name), M=\(batch), error=\(error)")
                }
                guard (batch * n..<batch * n + 64).allSatisfy({ value($0).isNaN }) else {
                    fatalError("Output guard")
                }
                var settlingDispatches = 0
                if CommandLine.arguments.contains("--settle-gpu") {
                    let settlingStart = DispatchTime.now().uptimeNanoseconds
                    while DispatchTime.now().uptimeNanoseconds - settlingStart < 100_000_000 {
                        _ = mlxDispatch()
                        _ = try dispatch()
                        settlingDispatches += 2
                    }
                }
                var wall: [Double] = []
                var gpu: [Double] = []
                var control: [Double] = []
                for trial in 0..<3 {
                    if paired && trial % 2 == 1 { control.append(mlxDispatch()) }
                    let time = try dispatch()
                    wall.append(time.0)
                    gpu.append(time.1)
                    if paired && trial % 2 == 0 { control.append(mlxDispatch()) }
                }
                results.append([
                    "backend": name, "weight_tile_rounding": kind >= 3 && kind != 6 ? "FP16" : "none",
                    "output_dtype": half ? "FP16" : "FP32", "M": batch, "N": n, "K": k,
                    "resident_wall_seconds": wall,
                    "gpu_seconds": gpu, "paired_control_wall_seconds": control,
                    "paired_order": paired ? "AB BA AB" : "unpaired", "static_full_tile": paired && kind == 3,
                    "relaxed_precision": relaxed,
                    "max_abs_error_vs_cpu": Double(error),
                    "control_numeric_relative_rmse": controlNumericRMSE ?? -1,
                    "settling_dispatches": settlingDispatches, "tile_N": tileN, "tile_M": tileM,
                ])
            }
        }
        if !paired {
            for batch in [1, 16, 128] {
                let x = MLXArray(Array(input.prefix(batch * k)), [batch, k])
                eval(x)
                var actual = quantizedMM(
                    x, packed, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
                eval(actual)
                Stream.gpu.synchronize()
                let values = actual.asType(.float32).asArray(Float.self)
                let reference = arrays["reference"]!.asArray(Float.self)
                var error: Float = 0
                var squared: Double = 0
                var energy: Double = 0
                for sample in 0..<batch {
                    for row in 0..<n {
                        let target = reference[sample * small.rows + row % small.rows]
                        let difference = values[sample * n + row] - target
                        error = max(error, abs(difference))
                        squared += Double(difference * difference)
                        energy += Double(target * target)
                    }
                }
                guard sqrt(squared / energy) < 0.003 else { fatalError("MLX dense-grid oracle failed") }
                var times: [Double] = []
                for _ in 0..<3 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    actual = quantizedMM(
                        x, packed, scales: scales, biases: biases, transpose: true, groupSize: 64, bits: 4)
                    eval(actual)
                    Stream.gpu.synchronize()
                    times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9)
                }
                results.append([
                    "backend": "project_pinned_mlx_q4_g64", "M": batch, "N": n, "K": k,
                    "output_dtype": "FP16", "resident_wall_seconds": times,
                    "max_abs_error_vs_float32_grid": Double(error),
                    "relative_rmse_vs_float32_grid": sqrt(squared / energy),
                ])
            }
        }
        let report: [String: Any] = [
            "scope":
                "Small runtime screen, repeated four-row artifact, resident buffers, one warmup and three samples; no end-to-end or full-model claim",
            "weight_codes": normal16
                ? (small.polynomialCoefficients == nil ? "fixed normal16 FP16 codebook" : "fixed cubic16 FP16 codebook")
                : "signed INT4",
            "device": device.name, "input_dtype": "FP16", "output_dtype": "specified per backend",
            "results": results,
        ]
        print(
            String(
                decoding: try JSONSerialization.data(
                    withJSONObject: report, options: [.prettyPrinted, .sortedKeys]),
                as: UTF8.self))
    }
}

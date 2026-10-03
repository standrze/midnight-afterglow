#if canImport(Metal)
    import Foundation
    import Metal

    /// Explicit experimental backends. Native INT4 requires macOS 26.4 and compatible hardware.
    public final class CompactInt4Metal {
        public enum Backend: String {
            case packed
            case nativeInt4
        }

        private let device: any MTLDevice
        private let queue: any MTLCommandQueue
        private let pipeline: any MTLComputePipelineState
        private let backend: Backend
        private let codeEncoding: CompactInt4Matrix.CodeEncoding
        private let scaleEncoding: CompactInt4Matrix.ScaleEncoding

        public init(
            backend: Backend, scaleEncoding: CompactInt4Matrix.ScaleEncoding = .e8m0,
            codeEncoding: CompactInt4Matrix.CodeEncoding = .signedInt4
        ) throws {
            guard codeEncoding != .normal16 || (backend == .packed && scaleEncoding == .e4m4) else {
                throw CompactInt4Error.invalid(
                    "Normal16 requires E4M4 scales and the packed backend; native INT4 cannot decode its codebook")
            }
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw CompactInt4Error.invalid("Metal unavailable")
            }
            let options = MTLCompileOptions()
            options.mathMode = .safe
            options.mathFloatingPointFunctions = .precise
            if backend == .nativeInt4 {
                guard #available(macOS 26.4, *), device.supportsFamily(.apple10) else {
                    throw CompactInt4Error.invalid("Native INT4 proof requires macOS 26.4 and Apple GPU family 10")
                }
                options.languageVersion = .version4_0
            }
            options.preprocessorMacros = [
                "CODE_NORMAL16": NSNumber(value: codeEncoding == .normal16 ? 1 : 0),
                "NATIVE_INT4": NSNumber(value: backend == .nativeInt4 ? 1 : 0),
                "SCALE_E3M4": NSNumber(value: scaleEncoding == .e3m4 ? 1 : 0),
                "SCALE_E4M4": NSNumber(value: scaleEncoding == .e4m4 ? 1 : 0),
            ]
            guard
                let resource = Bundle.module.url(
                    forResource: "CompactInt4", withExtension: "metal", subdirectory: "Kernels")
            else {
                throw CompactInt4Error.invalid("Missing Metal proof source")
            }
            let source = try String(contentsOf: resource, encoding: .utf8)
            let library = try device.makeLibrary(source: source, options: options)
            let name = backend == .nativeInt4 ? "compact_int4_native" : "compact_int4_packed"
            guard let function = library.makeFunction(name: name) else {
                throw CompactInt4Error.invalid("Missing Metal entry point")
            }
            self.pipeline = try device.makeComputePipelineState(function: function)
            self.device = device
            self.queue = queue
            self.backend = backend
            self.scaleEncoding = scaleEncoding
            self.codeEncoding = codeEncoding
        }

        /// Input is explicitly FP16; output is FP32. No implicit backend fallback.
        public func multiply(_ matrix: CompactInt4Matrix, input: [Float16], batch: Int) throws -> [Float] {
            let (inputCount, overflow) = batch.multipliedReportingOverflow(by: matrix.columns)
            let (outputCount, outputOverflow) = batch.multipliedReportingOverflow(by: matrix.rows)
            guard matrix.scaleEncoding == scaleEncoding, matrix.codeEncoding == codeEncoding, batch > 0, !overflow,
                !outputOverflow,
                input.count == inputCount,
                inputCount <= Int(UInt32.max), outputCount <= Int(UInt32.max),
                matrix.rows * matrix.columns <= Int(UInt32.max), input.allSatisfy(\.isFinite)
            else { throw CompactInt4Error.invalid("Invalid Metal input geometry") }
            let buffers = try [
                buffer(input), buffer(matrix.codes), buffer(matrix.scaleBytes),
                buffer(matrix.offsets), buffer(matrix.gains),
                buffer([Float](repeating: .nan, count: outputCount + 64)),
            ]
            guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw CompactInt4Error.invalid("Metal command allocation failed")
            }
            encoder.setComputePipelineState(pipeline)
            for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
            var shape = SIMD4<UInt32>(UInt32(batch), UInt32(matrix.rows), UInt32(matrix.columns), 0)
            encoder.setBytes(&shape, length: MemoryLayout.size(ofValue: shape), index: 6)
            let threads = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
            if backend == .nativeInt4 {
                encoder.dispatchThreadgroups(
                    MTLSize(width: (matrix.rows + 31) / 32, height: (batch + 15) / 16, depth: 1),
                    threadsPerThreadgroup: threads)
            } else {
                encoder.dispatchThreads(
                    MTLSize(width: outputCount, height: 1, depth: 1), threadsPerThreadgroup: threads)
            }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            guard command.status == .completed else {
                throw command.error ?? CompactInt4Error.invalid("Metal command did not complete")
            }
            let pointer = buffers[5].contents().bindMemory(to: Float.self, capacity: outputCount + 64)
            guard (outputCount..<(outputCount + 64)).allSatisfy({ pointer[$0].isNaN }) else {
                throw CompactInt4Error.invalid("Metal output guard overwritten")
            }
            let result = Array(UnsafeBufferPointer(start: pointer, count: outputCount))
            guard result.allSatisfy(\.isFinite) else { throw CompactInt4Error.invalid("Non-finite Metal output") }
            return result
        }

        private func buffer<T>(_ values: [T]) throws -> any MTLBuffer {
            guard
                let result = values.withUnsafeBytes({ bytes in
                    device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
                })
            else { throw CompactInt4Error.invalid("Metal buffer allocation failed") }
            return result
        }
    }
#endif

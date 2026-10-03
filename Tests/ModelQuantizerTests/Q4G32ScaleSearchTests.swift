import MLX
import MLXLMCommon
import XCTest

final class Q4G32ScaleSearchTests: XCTestCase {
    func testStoredGridErrorAndRowBatchPackingOnCPUAndMetal() {
        for device in [Device.cpu, Device.gpu] {
            Device.withDefaultDevice(device) {
                for groupSize in [32] {
                    for dtype in [DType.float32, .float16, .bfloat16] {
                        let width = groupSize * 2
                        var values = (0..<(17 * width)).map { index in
                            Float((index * 17) % 191 - 95) / 31
                        }
                        for index in 0..<width { values[index] = 0 }
                        for index in width..<(2 * width) { values[index] = 3 }
                        values[2 * width] = -13
                        values[3 * width - 1] = 17
                        let source = MLXArray(values).reshaped(17, width).asType(dtype)
                        let baseline = MLX.quantized(source, groupSize: groupSize, bits: 4)
                        let whole = q4AffineScaleSearchQuantized(source, groupSize: groupSize)
                        let batched = q4AffineScaleSearchQuantized(source, groupSize: groupSize, rowBatchSize: 3)
                        XCTAssertEqual(whole.weight.shape, [17, width / 8])
                        XCTAssertEqual(whole.scales.dtype, dtype)
                        XCTAssertEqual(whole.biases.dtype, dtype)
                        for (a, b) in [
                            (whole.weight, batched.weight), (whole.scales, batched.scales),
                            (whole.biases, batched.biases),
                        ] {
                            XCTAssertTrue(MLX.all(a .== b).item(Bool.self))
                        }
                        let candidate = MLX.dequantized(
                            whole.weight, scales: whole.scales.asType(.float32),
                            biases: whole.biases.asType(.float32), groupSize: groupSize, bits: 4)
                        let ordinary = MLX.dequantized(
                            baseline.wq, scales: baseline.scales.asType(.float32),
                            biases: baseline.biases!.asType(.float32), groupSize: groupSize, bits: 4)
                        let target = source.asType(.float32)
                        let candidateError = MLX.mean(MLX.square(target - candidate).reshaped(-1, groupSize), axis: -1)
                        let baselineError = MLX.mean(MLX.square(target - ordinary).reshaped(-1, groupSize), axis: -1)
                        XCTAssertTrue(MLX.all(candidateError .<= (baselineError + 0.0000001)).item(Bool.self))
                        XCTAssertTrue(MLX.all(MLX.isFinite(candidate)).item(Bool.self))
                        XCTAssertTrue(MLX.all(whole.weight[0] .== baseline.wq[0]).item(Bool.self))
                        XCTAssertTrue(MLX.all(whole.scales[0] .== baseline.scales[0]).item(Bool.self))
                        XCTAssertTrue(MLX.all(whole.biases[0] .== baseline.biases![0]).item(Bool.self))

                        let experts = source.reshaped(1, 17, width)
                        let expertResult = q4AffineScaleSearchQuantized(experts, groupSize: groupSize, rowBatchSize: 3)
                        XCTAssertEqual(expertResult.weight.shape, [1, 17, width / 8])
                        XCTAssertTrue(
                            MLX.all(expertResult.weight.reshaped(whole.weight.shape) .== whole.weight).item(Bool.self))
                    }
                }
            }
        }
    }
}

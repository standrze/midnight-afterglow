import AfterglowModelSupport
import MLX
import XCTest

final class Gemma4ProjectionInputRecorderTests: XCTestCase {
    func testDensePrefixIsCappedButAllPositionsAreCounted() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = Gemma4ProjectionInputTarget(path: "dense")
            let recorder = try Gemma4ProjectionInputRecorder(
                targets: [target], maximumPositions: 3, maximumRetainedBytes: 1024)
            recorder.observeDense(path: "ignored", input: MLXArray.zeros([1]))
            recorder.observeDense(path: "dense", input: MLXArray(Array(0..<8)).reshaped(2, 4).asType(.bfloat16))
            recorder.observeDense(path: "dense", input: MLXArray(Array(8..<20)).reshaped(3, 4).asType(.bfloat16))
            let result = try XCTUnwrap(recorder.finalize().first)
            XCTAssertEqual(result.observedPositions, 5)
            XCTAssertEqual(result.inputs.shape, [3, 4])
            XCTAssertEqual(result.inputs.asType(.float32).asArray(Float.self), (0..<12).map(Float.init))
        }
    }

    func testRoutedBroadcastCapturesOnlyActualSelectedExperts() throws {
        try Device.withDefaultDevice(.cpu) {
            let recorder = try Gemma4ProjectionInputRecorder(
                targets: [.init(path: "up", expert: 0), .init(path: "up", expert: 1)],
                maximumPositions: 2, maximumRetainedBytes: 1024)
            let ids = MLXArray([Int32(0), 1, 1, 1]).reshaped(1, 2, 2)
            // One token input broadcasts across its two selected experts.
            let input = MLXArray([Float(1), 2, 3, 4]).reshaped(1, 2, 1, 1, 2).asType(.bfloat16)
            recorder.observeRoutedProjection(path: "up", input: input, indices: ids, expertCount: 2)
            let results = try recorder.finalize()
            XCTAssertEqual(results.map(\.observedPositions), [1, 3])
            XCTAssertEqual(results[0].inputs.asType(.float32).asArray(Float.self), [1, 2])
            XCTAssertEqual(results[1].inputs.asType(.float32).asArray(Float.self), [1, 2, 3, 4])
        }
    }

    func testDownCaptureKeepsExpertSpecificPostActivationInputs() throws {
        try Device.withDefaultDevice(.cpu) {
            let recorder = try Gemma4ProjectionInputRecorder(
                targets: [.init(path: "down", expert: 1)], maximumPositions: 8,
                maximumRetainedBytes: 1024)
            let ids = MLXArray([Int32(0), 1, 1, 0]).reshaped(1, 2, 2)
            let input = MLXArray((0..<8).map(Float.init)).reshaped(1, 2, 2, 1, 2).asType(.bfloat16)
            recorder.observeRoutedProjection(path: "down", input: input, indices: ids, expertCount: 2)
            let result = try XCTUnwrap(recorder.finalize().first)
            XCTAssertEqual(result.inputs.asType(.float32).asArray(Float.self), [2, 3, 4, 5])
            XCTAssertEqual(result.observedPositions, 2)
        }
    }

    func testBudgetInvalidExpertAndNonfiniteCaptureFailExplicitly() throws {
        try Device.withDefaultDevice(.cpu) {
            let budget = try Gemma4ProjectionInputRecorder(
                targets: [.init(path: "dense")], maximumPositions: 4, maximumRetainedBytes: 31)
            budget.observeDense(path: "dense", input: MLXArray.ones([4, 4]).asType(.bfloat16))
            XCTAssertThrowsError(try budget.evaluatePending())
            let expert = try Gemma4ProjectionInputRecorder(
                targets: [.init(path: "up", expert: 1)], maximumPositions: 4, maximumRetainedBytes: 1024)
            expert.observeRoutedProjection(
                path: "up", input: MLXArray.ones([1, 1, 1, 1, 4]),
                indices: MLXArray([Int32(3)]).reshaped(1, 1, 1), expertCount: 2)
            XCTAssertThrowsError(try expert.finalize())
            let finite = try Gemma4ProjectionInputRecorder(
                targets: [.init(path: "dense")], maximumPositions: 4, maximumRetainedBytes: 1024)
            finite.observeDense(path: "dense", input: MLXArray([Float.nan]).reshaped(1, 1))
            XCTAssertThrowsError(try finite.finalize())
        }
    }

    func testMissingOrZeroCoverageCannotInventInputs() throws {
        try Device.withDefaultDevice(.cpu) {
            let recorder = try Gemma4ProjectionInputRecorder(
                targets: [.init(path: "up", expert: 1)], maximumPositions: 4, maximumRetainedBytes: 1024)
            recorder.observeRoutedProjection(
                path: "up", input: MLXArray.ones([1, 1, 1, 1, 4]),
                indices: MLXArray([Int32(0)]).reshaped(1, 1, 1), expertCount: 2)
            XCTAssertThrowsError(try recorder.finalize())
        }
    }
}

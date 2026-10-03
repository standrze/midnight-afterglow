#!/usr/bin/env python3
"""Pinned, idempotent native training overlays for Qwen3.5."""
import pathlib, sys
root = pathlib.Path(sys.argv[1])
p = root / 'Libraries/MLXLMCommon/GatedDelta.swift'
s = p.read_text()
marker = 'public enum GatedDeltaExecution {'
if marker not in s:
    anchor = 'public func gatedDeltaUpdate('
    if anchor not in s: raise SystemExit('Unsupported GatedDelta source')
    s = s.replace(anchor, 'public enum GatedDeltaExecution {\n    @TaskLocal public static var useMetalKernel = true\n    @TaskLocal public static var checkpointTraining = false\n}\n\n' + anchor, 1)
    anchor = 'if GatedDeltaKernelManager.shared.kernel != nil, Dk % 32 == 0 {'
    if anchor not in s: raise SystemExit('Unsupported GatedDelta kernel selector')
    s = s.replace(anchor, 'if GatedDeltaExecution.useMetalKernel, GatedDeltaKernelManager.shared.kernel != nil, Dk % 32 == 0 {', 1)
elif 'static var checkpointTraining' not in s:
    s = s.replace('@TaskLocal public static var useMetalKernel = true', '@TaskLocal public static var useMetalKernel = true\n    @TaskLocal public static var checkpointTraining = false',1)
if '// Afterglow checkpointed recurrent chunks' not in s:
    anchor = '    var ys = [MLXArray]()'
    insertion = '''    // Afterglow checkpointed recurrent chunks: all differentiable inputs are explicit.
    if GatedDeltaExecution.checkpointTraining {
        let chunk = checkpoint { arrays in
            var current = arrays[5]
            var output: [MLXArray] = []
            for index in 0..<arrays[0].dim(1) {
                let (y, next) = gatedDeltaStepOps(
                    q: arrays[0][0..., index], k: arrays[1][0..., index],
                    v: arrays[2][0..., index], g: arrays[3][0..., index],
                    beta: arrays[4][0..., index], state: current,
                    mask: arrays.count == 7 ? arrays[6][0..., index] : nil)
                output.append(y)
                current = next
            }
            return [MLX.stacked(output, axis: 1), current]
        }
        var outputs: [MLXArray] = []
        for start in stride(from: 0, to: T, by: 16) {
            let end = min(start + 16, T)
            var arrays = [q[0..., start..<end], k[0..., start..<end], v[0..., start..<end],
                g[0..., start..<end], beta[0..., start..<end], state]
            if let mask { arrays.append(mask[0..., start..<end]) }
            let result = chunk(arrays)
            outputs.append(result[0])
            state = result[1]
        }
        return (MLX.concatenated(outputs, axis: 1), state)
    }

'''
    if anchor not in s: raise SystemExit('Unsupported recurrence implementation')
    s = s.replace(anchor, insertion+anchor,1)
p.chmod(p.stat().st_mode | 0o200)
p.write_text(s)
# MLX-C provides checkpoint, but the pinned Swift API has no wrapper.
p = root.parent / 'mlx-swift/Source/MLX/Transforms.swift'
s = p.read_text()
if 'private final class AfterglowCheckpoint' not in s:
    s += '''

/// Recompute intermediate activations during backward evaluation.
public func checkpoint(_ function: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
    let owner = AfterglowCheckpoint(function)
    return { owner.call($0) }
}

private final class AfterglowCheckpoint {
    private var transformed: mlx_closure
    init(_ function: @escaping ([MLXArray]) -> [MLXArray]) {
        let original = new_mlx_closure(function)
        defer { mlx_closure_free(original) }
        var result = mlx_closure_new()
        let status = withEvalLock { mlx_checkpoint(&result, original) }
        precondition(status == 0, "Checkpoint transform failed")
        transformed = result
    }
    deinit { withEvalLock { mlx_closure_free(transformed) } }
    func call(_ arrays: [MLXArray]) -> [MLXArray] {
        withEvalLock {
            let inputs = new_mlx_vector_array(arrays)
            defer { mlx_vector_array_free(inputs) }
            var outputs = mlx_vector_array_new()
            defer { mlx_vector_array_free(outputs) }
            let status = mlx_closure_apply(&outputs, transformed, inputs)
            precondition(status == 0, "Checkpoint evaluation failed")
            return mlx_vector_array_values(outputs)
        }
    }
}
'''
    p.chmod(p.stat().st_mode | 0o200)
    p.write_text(s)

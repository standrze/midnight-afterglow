// Extracted for Wick from Midnight: Sources/ModelRunnerCore/LagunaFusedGatherSilu.swift
// Source snapshot SHA256: e34f54619362d23a0f14d116a4c67155553825b24e928645f96f363276492b7e
// Retains the Apache-2.0 license and original third-party attribution.
// This local copy is maintained independently; no Midnight checkout is required.

import MLX
import MLXLMCommon

/// Experimental primitive shared by the runner and its numerical benchmark.
///
/// The source follows MLX's MIT-licensed qmv_fast at revision 1f8e74e3f12f31365464a6867c6579f0e9b29d85; preserve its typed arithmetic.
@_spi(Benchmark) public enum LagunaFusedGatherSiluKernel {
    /// The caller must supply eight valid Laguna expert IDs from zero through 255.
    ///
    /// Shape checks do not read GPU values or synchronize execution.
    public static func supports(
        input: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        indices: MLXArray
    ) -> Bool {
        (input.shape == [1, 1, 2048] || input.shape == [1, 1, 1, 1, 2048])
            && input.dtype == .bfloat16
            && supportsParameters(weight: weight, scales: scales, biases: biases)
            && indices.shape == [1, 1, 8] && indices.dtype == .uint32
    }

    static func supportsParameters(weight: MLXArray, scales: MLXArray, biases: MLXArray) -> Bool {
        weight.shape == [256, 1024, 256] && weight.dtype == .uint32
            && scales.shape == [256, 1024, 32] && scales.dtype == .bfloat16
            && biases.shape == scales.shape && biases.dtype == .bfloat16
    }

    public static func callAsFunction(
        input: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        indices: MLXArray
    ) -> MLXArray? {
        #if os(macOS)
            guard Device.defaultDevice().deviceType == .gpu,
                supports(input: input, weight: weight, scales: scales, biases: biases, indices: indices)
            else { return nil }
            return kernel(
                [input, weight, scales, biases, indices],
                template: [("T", DType.bfloat16)], grid: (128 * 64, 8, 1),
                threadGroup: (64, 1, 1), outputShapes: [[1, 1, 8, 1, 512]],
                outputDTypes: [.bfloat16])[0]
        #else
            return nil
        #endif
    }

    #if os(macOS)
        // Contiguous model tensors are referenced directly. Unusual strided views
        // are copied by MLX at dispatch, without inspecting unstable lazy strides.
        private static let kernel = MLXFast.metalKernel(
            name: "laguna_affine_q4_g64_gather_gateup_silu_2048_512_v2",
            inputNames: ["x", "packed", "scales", "biases", "indices"],
            outputNames: ["output"], source: source, ensureRowContiguous: true)
    #endif

    public static let source = """
        const uint lane = thread_position_in_threadgroup.x % 32;
        const uint simd = thread_position_in_threadgroup.x / 32;
        const uint slot = threadgroup_position_in_grid.y;
        const uint expert = indices[slot];
        const uint first_row = threadgroup_position_in_grid.x * 4 + simd * 2;
        const device ushort* words = reinterpret_cast<const device ushort*>(packed);
        float gates[2] = {0.0f, 0.0f};
        float ups[2] = {0.0f, 0.0f};

        for (uint block = 0; block < 2048; block += 512) {
            const uint k0 = block + lane * 16;
            float xv[16];
            float sum = 0.0f;
            #pragma unroll
            for (uint i = 0; i < 16; i += 4) {
                float a = float(x[k0 + i]);
                float b = float(x[k0 + i + 1]);
                float c = float(x[k0 + i + 2]);
                float d = float(x[k0 + i + 3]);
                // Match qmv_fast's model-dtype expression before adding to FP32.
                sum += x[k0 + i] + x[k0 + i + 1] + x[k0 + i + 2] + x[k0 + i + 3];
                xv[i] = a;
                xv[i + 1] = b / 16.0f;
                xv[i + 2] = c / 256.0f;
                xv[i + 3] = d / 4096.0f;
            }
            #pragma unroll
            for (uint r = 0; r < 2; ++r) {
                const uint row = first_row + r;
                const uint gate_row = expert * 1024 + row;
                const uint up_row = gate_row + 512;
                const uint gate_word = gate_row * 512 + k0 / 4;
                const uint up_word = up_row * 512 + k0 / 4;
                float gate_dot = 0.0f;
                float up_dot = 0.0f;
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    const ushort g = words[gate_word + i];
                    const ushort u = words[up_word + i];
                    gate_dot += xv[4 * i] * (g & 0x000f)
                        + xv[4 * i + 1] * (g & 0x00f0)
                        + xv[4 * i + 2] * (g & 0x0f00)
                        + xv[4 * i + 3] * (g & 0xf000);
                    up_dot += xv[4 * i] * (u & 0x000f)
                        + xv[4 * i + 1] * (u & 0x00f0)
                        + xv[4 * i + 2] * (u & 0x0f00)
                        + xv[4 * i + 3] * (u & 0xf000);
                }
                const uint gg = gate_row * 32 + k0 / 64;
                const uint ug = up_row * 32 + k0 / 64;
                gates[r] += float(scales[gg]) * gate_dot + sum * float(biases[gg]);
                ups[r] += float(scales[ug]) * up_dot + sum * float(biases[ug]);
            }
        }
        #pragma unroll
        for (uint r = 0; r < 2; ++r) {
            float gate_sum = simd_sum(gates[r]);
            float up_sum = simd_sum(ups[r]);
            if (lane == 0) {
                // Stock gather materializes the two projections in model dtype.
                T gate = T(gate_sum);
                T up = T(up_sum);
                // PROJECTION_CAPTURE_POINT
                // Match unary_ops.h Sigmoid<T> and compiled.cpp's typed temporary
                // for every tape node: sigmoid, gate*sigmoid, then *up.
                auto tail = 1 / (1 + metal::exp(metal::abs(gate)));
                T probability = (gate < 0) ? tail : 1 - tail;
                T activated = gate * probability;
                output[slot * 512 + first_row + r] = activated * up;
            }
        }
        """
}

/// Bound only while tracing the candidate decode closure.
///
/// Parameter lookups retain the original arrays; there is no per-token lookup or dequantization.
struct LagunaFusedGateUpSiluBinding {
    private let weight: MLXArray
    private let scales: MLXArray
    private let biases: MLXArray
    private let down: SwitchLinear

    init?(_ layer: FusedGateUpSwitchGLU) {
        let leaves = Dictionary(uniqueKeysWithValues: layer.leafModules().flattened())
        guard let gateUp = leaves["gate_up_proj"] as? QuantizedSwitchLinear,
            type(of: gateUp) == QuantizedSwitchLinear.self,
            gateUp.mode == .affine, gateUp.bits == 4, gateUp.groupSize == 64,
            let down = leaves["down_proj"] as? SwitchLinear
        else { return nil }
        let parameters = Dictionary(uniqueKeysWithValues: gateUp.parameters().flattened())
        guard parameters["bias"] == nil,
            let weight = parameters["weight"], let scales = parameters["scales"],
            let biases = parameters["biases"],
            LagunaFusedGatherSiluKernel.supportsParameters(weight: weight, scales: scales, biases: biases)
        else { return nil }
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.down = down
    }

    func callAsFunction(_ input: MLXArray, _ indices: MLXArray) -> MLXArray? {
        guard
            let activated = LagunaFusedGatherSiluKernel.callAsFunction(
                input: input, weight: weight, scales: scales, biases: biases, indices: indices)
        else { return nil }
        return MLX.squeezed(down(activated, indices, sortedIndices: false), axis: -2)
    }
}

/// The experiment is limited to the existing compiled one-token decode path.
///
/// The ordinary MoE handles prefill, capture/calibration, and every fallback.
enum LagunaFusedGateUpSiluEligibility {
    static func allows(
        runtimeEnabled: Bool, useCompiledTail: Bool, useFusedRouter: Bool,
        training: Bool, hasCalibrationObserver: Bool
    ) -> Bool {
        #if os(macOS)
            runtimeEnabled && useCompiledTail && useFusedRouter && !training && !hasCalibrationObserver
        #else
            false
        #endif
    }
}

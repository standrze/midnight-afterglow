#include <metal_stdlib>
using namespace metal;

#if CODE_NORMAL16
constant ushort normal16_bits[16] = {49011u,48454u,48138u,47670u,47266u,46704u,45975u,44293u,11525u,13207u,13936u,14498u,14902u,15370u,15686u,16243u};
#endif
inline float decoded_code(uint nibble) {
#if CODE_NORMAL16
    return float(as_type<half>(normal16_bits[nibble]));
#else
    return float(int(nibble) - (nibble >= 8 ? 16 : 0));
#endif
}

inline float metadata(ushort bits) { return as_type<float>(uint(bits) << 16); }

inline float decoded_scale(uchar byte) {
#if SCALE_E3M4
    return as_type<float>((uint(byte & 128u) << 24) | ((uint((byte >> 4) & 7u) + 124u) << 23)
        | (uint(byte & 15u) << 19));
#elif SCALE_E4M4
    return as_type<float>(((uint(byte >> 4) + 120u) << 23) | (uint(byte & 15u) << 19));
#else
    return as_type<float>(byte == 0 ? 0x00400000u : uint(byte) << 23);
#endif
}

// Direct packed reference kernel, valid for decode and arbitrary batch sizes.
kernel void compact_int4_packed(
    const device half* input [[buffer(0)]],
    const device uchar* codes [[buffer(1)]],
    const device uchar* exponents [[buffer(2)]],
    const device ushort* offsets [[buffer(3)]],
    const device ushort* gains [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint4& shape [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    const uint M = shape.x, N = shape.y, K = shape.z;
    if (index >= M * N) return;
    const uint row = index % N, sample = index / N;
    float total = 0;
    for (uint group = 0; group < K; group += 32) {
        float dot = 0, sum = 0;
        for (uint k = group; k < group + 32; ++k) {
            const uint flat = row * K + k;
            const uint nibble = (codes[flat / 2] >> ((flat % 2) * 4)) & 15;
            const float code = decoded_code(nibble);
            const float x = float(input[sample * K + k]);
            dot += x * float(code);
            sum += x;
        }
        total += dot * decoded_scale(exponents[row * K / 32 + group / 32]);
        total += sum * metadata(offsets[row * K / 64 + group / 64]);
    }
    output[index] = total * metadata(gains[row]);
}

#if NATIVE_INT4
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// No weight expansion: native A16 x signed INT4 per G32, then scale/offset epilogue.
// This deliberately exposes G32 boundaries without requiring macOS 27 scale planes.
kernel void compact_int4_native(
    device half* input [[buffer(0)]],
    device uchar* codes [[buffer(1)]],
    const device uchar* exponents [[buffer(2)]],
    const device ushort* offsets [[buffer(3)]],
    const device ushort* gains [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint4& shape [[buffer(6)]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int, 2>, tensor_inline> x(
        input, dextents<int, 2>(shape.z, shape.x), array<int, 2>{1, int(shape.z)});
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> w(
        codes, dextents<int, 2>(shape.z, shape.y), array<int, 2>{1, int(shape.z)});
    constexpr auto descriptor = matmul2d_descriptor(
        16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> op;
    auto partial = op.get_destination_cooperative_tensor<decltype(x), decltype(w), float>();
    auto total = op.get_destination_cooperative_tensor<decltype(x), decltype(w), float>();
    for (ushort i = 0; i < total.get_capacity(); ++i) {
        if (total.is_valid_element(i)) total[i] = 0;
    }
    for (uint group = 0; group < shape.z; group += 32) {
        for (ushort i = 0; i < partial.get_capacity(); ++i) {
            if (partial.is_valid_element(i)) partial[i] = 0;
        }
        auto left = x.slice(group, tile.y * 16);
        auto right = w.slice(group, tile.x * 32);
        op.run(left, right, partial);
        for (ushort i = 0; i < total.get_capacity(); ++i) {
            if (total.is_valid_element(i)) {
                const auto coord = total.get_multidimensional_index(i);
                const uint sample = tile.y * 16 + coord[1];
                const uint row = tile.x * 32 + coord[0];
                if (sample < shape.x && row < shape.y) {
                    float sum = 0;
                    for (uint k = group; k < group + 32; ++k) sum += float(input[sample * shape.z + k]);
                    total[i] += partial[i] * decoded_scale(exponents[row * shape.z / 32 + group / 32]);
                    total[i] += sum * metadata(offsets[row * shape.z / 64 + group / 64]);
                }
            }
        }
    }
    for (ushort i = 0; i < total.get_capacity(); ++i) {
        if (total.is_valid_element(i)) {
            const auto coord = total.get_multidimensional_index(i);
            const uint sample = tile.y * 16 + coord[1];
            const uint row = tile.x * 32 + coord[0];
            if (sample < shape.x && row < shape.y) output[sample * shape.y + row] = total[i] * metadata(gains[row]);
        }
    }
}
#endif

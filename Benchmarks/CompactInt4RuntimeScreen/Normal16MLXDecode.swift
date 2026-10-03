import Foundation
import MLX

enum Normal16MLXDecode {
    static func kernel(rows: Int, book: [UInt16], lookup: Int = 0, polynomial: [Float]? = nil, simdGroups: Int = 1)
        -> MLXFast.MLXFastKernel
    {
        precondition(book.count == 16 && (0...3).contains(lookup) && [1, 2, 4].contains(simdGroups))
        for i in 0..<8 {
            precondition(book[i] == book[15 - i] | 0x8000)
        }
        let low = (0..<4).reduce(UInt64(0)) { $0 | UInt64(book[8 + $1]) << ($1 * 16) }
        let high = (0..<4).reduce(UInt64(0)) { $0 | UInt64(book[12 + $1]) << ($1 * 16) }
        let expression: String
        switch lookup {
        case 3:
            precondition(polynomial?.count == 2 && polynomial!.allSatisfy(\.isFinite))
            expression = """
                const float q = float(int(index) * 2 - 15);
                return float(half(q * (\(polynomial![0])f + \(polynomial![1])f * q * q)));
                """
        case 1:
            expression = """
                const uint m = index >= 8 ? index - 8 : 7 - index;
                const ulong packed = m < 4 ? \(low)ul : \(high)ul;
                return float(as_type<half>(ushort(ushort(packed >> (16 * (m & 3))) | ushort(index < 8 ? 0x8000 : 0))));
                """
        case 2:
            expression = "return float(simd_shuffle(lane_book, ushort(index)));"
        default:
            expression = "return float(as_type<half>(ag_normal_book[index]));"
        }
        let header = """
            constant ushort ag_normal_book[16] = {\(book.map(String.init).joined(separator: ","))};
            inline float ag_bf16(ushort v) { return as_type<float>(uint(v) << 16); }
            inline float ag_scale(uchar v) {
                return as_type<float>(((uint(v >> 4) + 120u) << 23) | (uint(v & 15u) << 19));
            }
            inline float ag_lookup(uint index, half lane_book) {
                \(expression)
            }
            """
        let source = """
            constexpr uint lanes_per_row = 32 / ROWS;
            const uint row = threadgroup_position_in_grid.x * ROWS * SG + simdgroup_index_in_threadgroup * ROWS + thread_index_in_simdgroup / lanes_per_row;
            const uint pack = thread_index_in_simdgroup % lanes_per_row;
            const half lane_book = as_type<half>(ag_normal_book[thread_index_in_simdgroup & 15]);
            const device uint* words = reinterpret_cast<const device uint*>(codes);
            float total = 0;
            for (uint group = 0; group < K; group += lanes_per_row * 8) {
                const uint k = group + pack * 8;
                const bool valid = row < N && k < K;
                const uint word = valid ? words[row * K / 8 + k / 8] : 0;
                float dot = 0, sum = 0;
            #pragma clang loop unroll(full)
                for (uint i = 0; i < 8; ++i) {
                    // Every lane executes the lookup, including partial rows/K.
                    const float code = ag_lookup((word >> (4 * i)) & 15u, lane_book);
                    const float x = valid ? float(input[k + i]) : 0;
                    dot += x * code;
                    sum += x;
                }
                if (valid) {
                    total += dot * ag_scale(scales[row * K / 32 + k / 32]);
                    total += sum * ag_bf16(offsets[row * K / 64 + k / 64]);
                }
            }
            #pragma clang loop unroll(full)
            for (uint shift = 1; shift < lanes_per_row; shift *= 2) total += simd_shuffle_xor(total, shift);
            if (pack == 0 && row < N) output[row] = total * ag_bf16(gains[row]);
            """
        return MLXFast.metalKernel(
            name: "ag_normal16_decode_r\(rows)_lookup\(lookup)_sg\(simdGroups)",
            inputNames: ["input", "codes", "scales", "offsets", "gains"], outputNames: ["output"],
            source: source, header: header)
    }
}

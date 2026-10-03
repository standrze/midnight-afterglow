import CompactInt4
import Foundation
import MLX

enum Normal16MLXPrefill {
    static func kernel(tileM: Int, shader: String) throws -> MLXFast.MLXFastKernel {
        guard let headerEnd = shader.range(of: "// Direct packed reference kernel"),
            let entry = shader.range(of: "kernel void compact_int4_nax("),
            let bodyStart = shader.range(of: ") {", range: entry.upperBound..<shader.endIndex),
            let bodyEnd = shader.range(of: "\n}\n\n#endif", range: bodyStart.upperBound..<shader.endIndex)
        else { throw CompactInt4Error.invalid("Missing research NAX source") }
        let header = """
            #define NATIVE_INT4 1
            #define NORMAL16 1
            #define OUTPUT_HALF 1
            #define SCALE_E3M4 0
            #define SCALE_E4M4 1
            #define PACKED_NAX 1
            #define NAX_BM BM
            #define NAX_BN 64
            \(shader[..<headerEnd.lowerBound])
            """
        let source = """
            const uint4 shape = uint4(M, N, K, 0);
            const uint2 tile = threadgroup_position_in_grid.xy;
            const uint lid = thread_index_in_threadgroup;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            \(shader[bodyStart.upperBound..<bodyEnd.lowerBound])
            """
        return MLXFast.metalKernel(
            name: "ag_normal16_prefill_bm\(tileM)",
            inputNames: ["input", "codes", "scales", "offsets", "gains"], outputNames: ["output"],
            source: source, header: header)
    }
}

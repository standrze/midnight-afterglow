#!/usr/bin/env python3
"""Independent CPU-only format/control check for a bounded MATRIX.json fixture.

This does not measure generated model quality or kernel speed. FNV checksum and
strict artifact validation are covered by the Swift reader; this deliberately
independent parser checks the arithmetic and storage accounting against MLX.
"""
import argparse
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', type=Path)
    parser.add_argument('artifact', type=Path)
    args = parser.parse_args()
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    source = json.loads(args.matrix.read_text())
    rows, cols = source['rows'], source['columns']
    weights = mx.array(source['weights'], dtype=mx.float32).reshape(rows, cols)
    data = args.artifact.read_bytes()
    if data[:8] != b'AGQ4E001' or struct.unpack_from('<II', data, 8) != (rows, cols):
        raise ValueError('Artifact/fixture mismatch')
    count = rows * cols
    if len(data) != 24 + count // 2 + count // 16 + rows * 2:
        raise ValueError('Artifact length mismatch')
    offset = 16
    codes = mx.array(list(data[offset:offset + count // 2]), dtype=mx.uint8)
    codes = mx.stack([codes & 15, codes >> 4], axis=-1).reshape(rows, cols).astype(mx.float32)
    codes = mx.where(codes >= 8, codes - 16, codes)
    offset += count // 2
    scales = mx.power(2., mx.array(list(data[offset:offset + count // 32]), dtype=mx.float32) - 127)
    offset += count // 32
    def bf16_values(n):
        nonlocal offset
        values = [struct.unpack('<f', struct.pack('<I', struct.unpack_from('<H', data, offset + i * 2)[0] << 16))[0] for i in range(n)]
        offset += n * 2
        return mx.array(values, dtype=mx.float32)
    biases = bf16_values(count // 64).reshape(rows, cols // 64)
    gains = bf16_values(rows).reshape(rows, 1)
    restored = gains * (codes * mx.repeat(scales.reshape(rows, cols // 32), 32, axis=1) + mx.repeat(biases, 64, axis=1))
    # Stored-grid controls, with BF16 metadata in both ordinary Q4 formats.
    def native_grid(group):
        packed, s, b = mx.quantize(weights.astype(mx.bfloat16), group_size=group, bits=4)
        q = ((packed[..., None] >> mx.arange(0, 32, 4, dtype=mx.uint32)) & 15).reshape(rows, cols).astype(mx.float32)
        return q * mx.repeat(s.astype(mx.float32), group, axis=1) + mx.repeat(b.astype(mx.float32), group, axis=1)
    grouped = weights.reshape(rows, cols // 32, 32)
    sym_scale = (mx.max(mx.abs(grouped), axis=-1, keepdims=True) / 7).astype(mx.bfloat16).astype(mx.float32)
    safe = mx.where(sym_scale != 0, sym_scale, 1)
    symmetric = (mx.clip(mx.round(grouped / safe), -8, 7) * sym_scale).reshape(rows, cols)
    energy = float(mx.sum(weights ** 2).item())
    result = {'source': source['source'], 'slice_sha256': source.get('slice_sha256'),
              'rows': rows, 'columns': cols, 'metrics': {},
              'model_quality_measured': False, 'speedup_measured': False,
              'scope': 'bounded weight slice; FP32 stored-grid reconstruction, no activation calibration'}
    for name, candidate, bpw in [('compact_int4', restored, 4.5 + 16 / cols),
                                  ('mlx_affine_q4_g64', native_grid(64), 4.5),
                                  ('symmetric_q4_g32', symmetric, 4.5)]:
        error = float(mx.sum((weights - candidate)**2).item())
        result['metrics'][name] = {'relative_squared_error': error / energy if energy else 0., 'bits_per_weight': bpw}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

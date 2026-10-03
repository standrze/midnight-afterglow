#!/usr/bin/env python3
"""Bounded real-input affine GPTQ-style probe; never a full-model quality benchmark.

Reads selected BF16 embedding/projection rows directly from indexed safetensors.
GPTQ equations: https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py
The adaptation uses native MLX affine parameters rounded to the source dtype,
contiguous groups, a full input covariance, and block error compensation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import tempfile
import time


def digest_json(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


class SelectedReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        self.evidence = []

    def info(self, key):
        shard = self.index[key]
        if Path(shard).is_absolute() or ".." in Path(shard).parts:
            raise ValueError("invalid shard path")
        if shard not in self.headers:
            with (self.root / shard).open("rb") as f:
                length = struct.unpack("<Q", f.read(8))[0]
                if length > 64 * 1024 * 1024:
                    raise ValueError("oversized header")
                if length > (self.root / shard).stat().st_size - 8:
                    raise ValueError("header exceeds source file")
                self.headers[shard] = (8 + length, json.loads(f.read(length)))
        offset, header = self.headers[shard]
        return self.root / shard, offset, header[key]

    def read(self, key, rows=None):
        import mlx.core as mx
        path, offset, info = self.info(key)
        sizes = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4}
        size = sizes[info["dtype"]]
        shape = info["shape"]
        start, end = info["data_offsets"]
        if not shape or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("invalid tensor shape")
        if (type(start) is not int or type(end) is not int or start < 0 or end < start
                or offset + end > path.stat().st_size):
            raise ValueError("tensor offsets outside source file")
        if math.prod(shape) * size != end - start:
            raise ValueError("tensor size mismatch")
        with path.open("rb") as f:
            if rows is None:
                if end - start > 32 * 1024 * 1024:
                    raise ValueError("probe prohibits large unselected tensors")
                f.seek(offset + start)
                payload = f.read(end - start)
            else:
                if len(shape) != 2 or len(set(rows)) != len(rows):
                    raise ValueError("expected unique matrix rows")
                width = shape[1] * size
                if not rows or len(rows) * width > 32 * 1024 * 1024:
                    raise ValueError("selected payload exceeds probe limit")
                payload = bytearray()
                for row in rows:
                    if row < 0 or row >= shape[0]:
                        raise ValueError("row outside matrix")
                    f.seek(offset + start + row * width)
                    chunk = f.read(width)
                    if len(chunk) != width:
                        raise ValueError("truncated tensor")
                    payload.extend(chunk)
                shape = [len(rows), shape[1]]
        if len(payload) != math.prod(shape) * size:
            raise ValueError("truncated tensor")
        evidence = {"key": key, "shape": shape, "dtype": info["dtype"],
                    "rows_sha256": digest_json(rows), "payload_sha256": hashlib.sha256(payload).hexdigest()}
        self.evidence.append(evidence)
        header = json.dumps({"selected": {"dtype": info["dtype"], "shape": shape,
                                         "data_offsets": [0, len(payload)]}}).encode()
        header += b" " * ((-len(header)) % 8)
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(payload)
            f.flush()
            value = mx.load(f.name)["selected"]
            mx.eval(value)
        return value


def corpus(path, tokenizer, samples, tokens_per_sample):
    data = Path(path).read_bytes()
    sequences = []
    source_fingerprints = []
    for line in data.decode().splitlines():
        record = json.loads(line)
        text = record.get("text", "")
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            continue
        source_fingerprints.append(digest_json(ids))
        sequences.append(ids[:tokens_per_sample])
        if len(sequences) == samples:
            break
    if not sequences:
        raise ValueError("empty corpus")
    return [v for sequence in sequences for v in sequence], {
        "path": str(Path(path).resolve()), "file_sha256": hashlib.sha256(data).hexdigest(),
        "tokenization": "tokenizers.Tokenizer.encode(add_special_tokens=False); per-record prefix",
        "samples": len(sequences), "tokens": sum(map(len, sequences)),
        "unique_token_ids": len({token for sequence in sequences for token in sequence}),
        "overlap_check_scope": "Exact full and selected sample sequences only; shared substrings/tokens are allowed",
        "full_sample_token_sha256": source_fingerprints,
        "selected_sample_token_sha256": [digest_json(s) for s in sequences],
    }


def unpack(quantized, group):
    import mlx.core as mx
    packed, scales, biases = quantized
    shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
    codes = ((packed[..., None] >> shifts) & 15).reshape(packed.shape[0], -1)
    grid = codes.reshape(scales.shape[0], -1, group).astype(mx.float32)
    return (grid * scales.astype(mx.float32)[..., None]
            + biases.astype(mx.float32)[..., None]).reshape(codes.shape)


def packed_codes(codes):
    import mlx.core as mx
    shifts = mx.arange(0, 32, 4, dtype=mx.uint32)
    return mx.sum(codes.reshape(codes.shape[0], -1, 8) << shifts, axis=-1).astype(mx.uint32)


def require_finite(value, label):
    import mlx.core as mx
    if not bool(mx.all(mx.isfinite(value)).item()):
        raise ValueError(f"non-finite {label}")


def nearest_code(column, scale, bias):
    import mlx.core as mx
    valid = mx.abs(scale) > 0
    safe = mx.where(valid, scale, mx.ones_like(scale))
    return mx.where(valid, mx.clip(mx.round((column - bias) / safe), 0, 15), 0).astype(mx.uint32)


def inverse_factor(inputs, damping):
    import mlx.core as mx
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive and finite")
    require_finite(inputs, "Hessian inputs")
    h = inputs.T @ inputs / inputs.shape[0]
    require_finite(h, "Hessian")
    diagonal = mx.arange(h.shape[0])
    # A zero-coverage input column has zero covariance with every other column.
    # Give it an independent positive diagonal while preserving its source weights.
    h[diagonal, diagonal] = mx.where(mx.diag(h) == 0, 1, mx.diag(h))
    damp = damping * mx.mean(mx.diag(h))
    h[diagonal, diagonal] += damp
    lower = mx.linalg.cholesky(h)
    inverse = mx.linalg.cholesky_inv(lower)
    factor = mx.linalg.cholesky(inverse, upper=True)
    mx.eval(factor)
    require_finite(factor, "inverse Hessian factor")
    return factor, float(damp.item())


def gptq_style(weight, factor, group, block_size=128):
    import mlx.core as mx
    if weight.shape[1] % block_size or block_size % group:
        raise ValueError("block and group geometry mismatch")
    if factor.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Hessian/weight geometry mismatch")
    require_finite(weight, "source weight")
    require_finite(factor, "inverse Hessian factor")
    if not bool(mx.all(mx.diag(factor) > 0).item()):
        raise ValueError("inverse Hessian factor must have positive diagonal")
    corrected = weight.astype(mx.float32)
    codes = mx.zeros(weight.shape, dtype=mx.uint32)
    scales, biases = [], []
    for start in range(0, weight.shape[1], block_size):
        end = start + block_size
        block = corrected[:, start:end]
        errors = mx.zeros_like(block)
        for j in range(block_size):
            if j % group == 0:
                # Native affine parameters use the actual stored BF16 grid.
                _, s, b = mx.quantize(block[:, j:j + group].astype(weight.dtype),
                                      group_size=group, bits=4)
                scales.append(s)
                biases.append(b)
                scale, bias = s.astype(mx.float32), b.astype(mx.float32)
            column = block[:, j:j + 1]
            q = nearest_code(column, scale, bias)
            reconstruction = q.astype(mx.float32) * scale + bias
            error = (column - reconstruction) / factor[start + j, start + j]
            codes[:, start + j:start + j + 1] = q
            errors[:, j:j + 1] = error
            block[:, j:] -= error @ factor[start + j:start + j + 1, start + j:end]
            mx.eval(block, errors, codes)
        corrected[:, end:] -= errors @ factor[start:end, end:]
        mx.eval(corrected)
    result = packed_codes(codes), mx.concatenate(scales, axis=1), mx.concatenate(biases, axis=1)
    mx.eval(result)
    for value in result:
        require_finite(value, "quantized result")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--tokens-per-sample", type=int, default=512)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--damping", type=float, default=0.01)
    parser.add_argument("--reference", action="append", default=[], metavar="LABEL=PATH")
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    if not (1 <= args.samples <= 32 and 1 <= args.tokens_per_sample <= 512 and 1 <= args.rows <= 512):
        raise ValueError("probe limits exceeded")
    if not math.isfinite(args.damping) or args.damping <= 0:
        raise ValueError("damping must be positive and finite")
    import mlx.core as mx
    import tokenizers
    from tokenizers import Tokenizer
    mx.set_default_device(mx.cpu)
    mx.set_memory_limit(2 * 1024**3)
    mx.set_cache_limit(128 * 1024**2)
    start = time.perf_counter()
    source = SelectedReader(args.source)
    config_data = (source.root / "config.json").read_bytes()
    config = json.loads(config_data)
    if config.get("model_type") != "laguna":
        raise ValueError("only exact Laguna first-layer input path supported")
    tokenizer_data = (source.root / "tokenizer.json").read_bytes()
    tokenizer = Tokenizer.from_str(tokenizer_data.decode())
    train_ids, train_meta = corpus(args.calibration, tokenizer, args.samples, args.tokens_per_sample)
    dev_ids, dev_meta = corpus(args.development, tokenizer, args.samples, args.tokens_per_sample)
    for field in ["full_sample_token_sha256", "selected_sample_token_sha256"]:
        if set(train_meta[field]) & set(dev_meta[field]):
            raise ValueError("calibration/development sample overlap")
    unique = sorted(set(train_ids + dev_ids))
    embedding = source.read("model.embed_tokens.weight", unique)
    norm = source.read("model.layers.0.input_layernorm.weight")
    key = "model.layers.0.self_attn.q_proj.weight"
    _, _, info = source.info(key)
    rows = [i * info["shape"][0] // args.rows for i in range(args.rows)]
    weight = source.read(key, rows)
    if embedding.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16 or norm.dtype != mx.bfloat16:
        raise ValueError("probe requires BF16 source tensors")
    positions = {value: index for index, value in enumerate(unique)}
    def inputs(ids):
        selected = embedding[mx.array([positions[t] for t in ids], dtype=mx.int32)]
        return mx.fast.rms_norm(selected, norm, config["rms_norm_eps"])
    train, dev = inputs(train_ids), inputs(dev_ids)
    mx.eval(train, dev)
    require_finite(train, "calibration input")
    require_finite(dev, "development input")
    train32, dev32, w32 = train.astype(mx.float32), dev.astype(mx.float32), weight.astype(mx.float32)
    exact_train, exact_dev = train32 @ w32.T, dev32 @ w32.T
    mx.eval(exact_train, exact_dev)
    extraction_seconds = time.perf_counter() - start
    h_start = time.perf_counter()
    factor, absolute_damping = inverse_factor(train32, args.damping)
    h_seconds = time.perf_counter() - h_start
    results = []
    def measure(label, q, group, seconds):
        reconstruction = unpack(q, group)
        train_error, dev_error = train32 @ (reconstruction - w32).T, dev32 @ (reconstruction - w32).T
        require_finite(reconstruction, "stored affine reconstruction")
        require_finite(train_error, "calibration output error")
        require_finite(dev_error, "development output error")
        record = {"label": label, "group_size": group, "bits": 4,
                  "parameter_dtype": str(q[1].dtype), "quantization_seconds": seconds,
                  "tensor_bytes": sum(a.nbytes for a in q),
                  "weight_mse": float(mx.mean((reconstruction - w32)**2).item())}
        for split, error, exact in [("calibration", train_error, exact_train), ("development", dev_error, exact_dev)]:
            mse = float(mx.mean(error**2).item())
            record[split + "_output_mse"] = mse
            reference_power = float(mx.mean(exact**2).item())
            if not math.isfinite(reference_power) or reference_power <= 0:
                raise ValueError("reference output power must be positive and finite")
            record[split + "_output_relative_mse"] = mse / reference_power
        # Ensure packing reconstructs the same grid as MLX dequantization with FP32 metadata.
        native = mx.dequantize(q[0], q[1].astype(mx.float32), q[2].astype(mx.float32), group_size=group, bits=4)
        if not bool(mx.array_equal(native, reconstruction).item()):
            raise ValueError("packed reconstruction does not match native affine dequantization")
        results.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    for group in [64, 128]:
        t = time.perf_counter()
        q = mx.quantize(weight, group_size=group, bits=4)
        mx.eval(q)
        measure("native-standard-cpu", q, group, time.perf_counter() - t)
        t = time.perf_counter()
        q = gptq_style(weight, factor, group)
        measure("native-affine-gptq-style-cpu", q, group, time.perf_counter() - t)
    references = []
    for value in args.reference:
        label, path = value.split("=", 1)
        reader = SelectedReader(path)
        module = "language_model.model.layers.0.self_attn.q_proj"
        q = tuple(reader.read(module + "." + suffix, rows) for suffix in ["weight", "scales", "biases"])
        group = weight.shape[1] // q[1].shape[1]
        measure(label, q, group, None)
        references.append({"label": label, "path": str(reader.root), "selected_tensors": reader.evidence})
    report = {"schema": "laguna-native-affine-gptq-projection-probe-v1",
              "scope": "128-row subset by default of real Laguna layer0 q_proj with complete input width; local output-error proxy only",
              "algorithm": "blocked GPTQ-style affine error compensation, block128, contiguous fixed groups, no activation order, native BF16 stored metadata",
              "implementation_reference": "https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": str(mx.default_device()), "mlx_version": mx.__version__, "tokenizers_version": tokenizers.__version__,
              "source": str(source.root), "config_sha256": hashlib.sha256(config_data).hexdigest(),
              "tokenizer_sha256": hashlib.sha256(tokenizer_data).hexdigest(), "selected_tensors": source.evidence,
              "projection_rows": rows, "calibration": train_meta, "development": dev_meta,
              "hessian_shape": list(factor.shape), "hessian_float32_bytes": factor.nbytes,
              "normalized_input_dtype": str(train.dtype),
              "empirical_covariance_rank_upper_bound": min(weight.shape[1], len(set(train_ids))),
              "relative_damping": args.damping, "absolute_damping": absolute_damping,
              "extraction_seconds": extraction_seconds, "hessian_factor_seconds": h_seconds,
              "total_seconds": time.perf_counter() - start, "mlx_peak_bytes": mx.get_peak_memory(),
              "process_max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "references": references, "results": results,
              "limitations": ["No end-to-end model loss or generation test", "No routed-expert coverage",
                              "Adapted algorithm; not bit-exact official GPTQ",
                              "Repeated token inputs can make empirical covariance rank-deficient before damping",
                              "Duplicate sequence checks do not establish absence of substring or token overlap",
                              "Evenly distributed projection rows are a structured channel subset",
                              "First-layer input depends only on token embedding and RMSNorm; not contextual downstream activation",
                              "References may have been converted on GPU; native standard and GPTQ-style probe use CPU",
                              "Selected payload fingerprints establish probe reproducibility, not whole-checkpoint identity"]}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # Exclusive creation also protects an output that appears after the early guard.
    with output_path.open("x") as destination:
        destination.write(encoded)


if __name__ == "__main__":
    main()

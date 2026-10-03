#!/usr/bin/env python3
"""Read safetensors headers only; plan >=4-bit Gemma grouping without converting weights."""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
}
FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}


def read_headers(root):
    """Validate header geometry/offsets, never read a tensor payload."""
    tensors, fingerprints = {}, {}
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("no local safetensors shards")
    for path in shards:
        size = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors prefix: {path.name}")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= min(100_000_000, size - 8):
                raise ValueError(f"invalid safetensors header length: {path.name}")
            raw = source.read(length)
        header = json.loads(raw)
        fingerprints[path.name] = {"header_sha256": hashlib.sha256(raw).hexdigest(), "file_bytes": size}
        ranges = []
        for name, item in header.items():
            if name == "__metadata__":
                continue
            shape, dtype, offsets = item["shape"], item["dtype"], item["data_offsets"]
            if (name in tensors or dtype not in DTYPE_BYTES or not isinstance(shape, list)
                    or any(type(n) is not int or n < 0 for n in shape)
                    or len(offsets) != 2 or any(type(n) is not int for n in offsets)):
                raise ValueError(f"unsupported or duplicate tensor header: {name}")
            start, end = offsets
            count = math.prod(shape) * DTYPE_BYTES[dtype]
            if start < 0 or end - start != count or end > size - 8 - length:
                raise ValueError(f"invalid tensor payload extent: {name}")
            tensors[name] = {"shape": shape, "dtype": dtype, "bytes": count}
            if end > start:
                ranges.append((start, end))
        ranges.sort()
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError(f"overlapping tensor payload extents: {path.name}")
    return tensors, fingerprints


def analyze(configuration, tensors):
    text = configuration.get("text_config", configuration)
    if text.get("model_type") not in {"gemma3", "gemma3_text", "gemma4", "gemma4_text"}:
        raise ValueError("this planner supports Gemma 3/4 only")
    primary, alias = configuration.get("quantization"), configuration.get("quantization_config")
    if primary is not None and alias is not None and primary != alias:
        raise ValueError("quantization and quantization_config disagree")
    quantization = primary if primary is not None else alias
    if not isinstance(quantization, dict) or "config_groups" in quantization:
        raise ValueError("requires native MLX affine metadata; compressed-tensors/QAT grids require explicit import")
    if quantization.get("quant_method") or quantization.get("mode", "affine") != "affine":
        raise ValueError("requires native MLX affine packed weights; foreign formats are not interchangeable")
    # Reject a below-four-bit declaration even if it names a module absent from these files.
    for geometry in [quantization] + [v for v in quantization.values() if isinstance(v, dict)]:
        if type(geometry.get("bits", 4)) is not int or geometry.get("bits", 4) not in {4, 5, 6, 8}:
            raise ValueError("the minimum bit width is four; supported affine widths are 4, 5, 6, 8")
    activation_name = text.get("dtype", text.get("torch_dtype", configuration.get("dtype")))
    activation_dtype = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}.get(activation_name)
    experts = text.get("num_experts") or 0
    top_k = text.get("top_k_experts") or 0
    if experts and not 0 < top_k <= experts:
        raise ValueError("invalid top_k_experts/num_experts for active-byte estimate")

    def active_fraction(name, tensor):
        if ".experts." in name and experts and len(tensor["shape"]) >= 3:
            if tensor["shape"][0] != experts:
                raise ValueError(f"expert tensor shape does not match config: {name}")
            return top_k / experts
        return 1.0

    total_bytes = sum(t["bytes"] for t in tensors.values())
    active_bytes = sum(t["bytes"] * active_fraction(n, t) for n, t in tensors.items())
    rows, warnings, metadata_dtypes = [], [], collections.Counter()
    used_metadata = set()
    for name, weight in sorted(tensors.items()):
        if not name.endswith(".weight") or weight["dtype"] != "U32":
            continue
        module = name.removesuffix(".weight")
        override = quantization.get(module, {})
        if not isinstance(override, dict):
            raise ValueError(f"packed module disabled or has invalid geometry: {module}")
        bits = override.get("bits", quantization.get("bits", 4))
        group_size = override.get("group_size", quantization.get("group_size", 64))
        mode = override.get("mode", quantization.get("mode", "affine"))
        if mode != "affine" or type(group_size) is not int or group_size not in {32, 64, 128}:
            raise ValueError(f"unsupported affine module geometry: {module}")
        if len(weight["shape"]) not in {2, 3} or weight["shape"][-1] * 32 % bits:
            raise ValueError(f"invalid packed matrix geometry: {module}")
        width = weight["shape"][-1] * 32 // bits
        expected = weight["shape"][:-1] + [width // group_size]
        scales = tensors.get(module + ".scales")
        biases = tensors.get(module + ".biases")
        if (width % group_size or scales is None or biases is None
                or scales["shape"] != expected or biases["shape"] != expected
                or scales["dtype"] not in FLOAT_DTYPES or biases["dtype"] not in FLOAT_DTYPES):
            raise ValueError(f"scale/bias shape or dtype disagrees with declared packed grid: {module}")
        used_metadata.update((module + ".scales", module + ".biases"))
        metadata_dtypes[(scales["dtype"], biases["dtype"])] += 1
        aligned_dtype = activation_dtype is not None and scales["dtype"] == biases["dtype"] == activation_dtype
        if not aligned_dtype:
            warnings.append(f"{module}: activation dtype {activation_dtype or 'unknown'} versus "
                            f"scale/bias {scales['dtype']}/{biases['dtype']}; audit runtime promotion")
        protected = module.endswith(("embed_tokens", "lm_head")) or ".router." in module
        compatible = bits == 4 and group_size == 64 and width % 128 == 0 and not protected
        would_lose_tail = (compatible and width >= 512 and width % 512 != 0
                           and weight["shape"][-2] % 8 == 0
                           and aligned_dtype and activation_dtype in {"F16", "BF16"})
        saving = (scales["bytes"] + biases["bytes"]) // 2 if compatible else 0
        fraction = active_fraction(name, weight)
        rows.append({
            "module": module, "bits": bits, "group_size": group_size, "input_width": width,
            "packed_shape": weight["shape"], "scale_dtype": scales["dtype"], "bias_dtype": biases["dtype"],
            "activation_dtype_aligned": aligned_dtype, "protected_embedding_head_or_router": protected,
            "g128_compatible": compatible, "g128_loses_opt_in_g64_tail": would_lose_tail,
            "g128_saved_bytes": saving, "g128_saved_active_bytes": saving * fraction,
            "retain_tail_plan_group_size": 128 if compatible and not would_lose_tail else group_size,
        })
    if not rows:
        raise ValueError("no native packed affine weight modules found")
    orphaned = [n for n in tensors if n.endswith((".scales", ".biases")) and n not in used_metadata]
    if orphaned:
        raise ValueError(f"quantization metadata has no corresponding packed weight: {orphaned[0]}")
    plans = {}
    for plan, selected in {
        "all_compatible_g128": [r for r in rows if r["g128_compatible"]],
        "retain_g64_tail_g128": [r for r in rows if r["g128_compatible"] and not r["g128_loses_opt_in_g64_tail"]],
    }.items():
        saved = sum(r["g128_saved_bytes"] for r in selected)
        active_saved = sum(r["g128_saved_active_bytes"] for r in selected)
        plans[plan] = {
            "changed_modules": len(selected), "saved_bytes": saved, "saved_active_bytes": active_saved,
            "projected_tensor_bytes": total_bytes - saved, "projected_active_bytes": active_bytes - active_saved,
            "ideal_active_byte_speedup_ceiling": active_bytes / (active_bytes - active_saved),
        }
    return {
        "format_version": 1,
        "scope": "Header-only >=4-bit geometry plan. No payload/finite/grid/model-quality validation; no conversion.",
        "active_byte_assumption": "One token, top-k selected experts, all other tensors read once; excludes KV, "
                                  "activation traffic, actual caching and launch overhead. Not measured bandwidth.",
        "conversion_status": "Wick supports opt-in source-based Gemma 3/4 conversion with --gemma-group-policy "
                             "and --g128-module; an unquantized BF16/FP16 source is required. "
                             "This header-only planner does not convert weights or establish quality/speed. "
                             "Do not rewrite config alone or requantize an installed Q4 checkpoint.",
        "activation_dtype_from_config": activation_dtype, "current_tensor_bytes": total_bytes,
        "estimated_current_active_bytes": active_bytes, "plans": plans,
        "metadata_dtype_counts": {f"{s}/{b}": n for (s, b), n in sorted(metadata_dtypes.items())},
        "warnings": warnings, "modules": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, help="New report file; defaults to stdout")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists")
    try:
        config_data = (args.model / "config.json").read_bytes()
        tensors, headers = read_headers(args.model)
        report = analyze(json.loads(config_data), tensors)
        report.update(model=str(args.model.resolve()), config_sha256=hashlib.sha256(config_data).hexdigest(),
                      safetensors_headers=headers, fingerprint_scope="Header hashes only, not weight payload identity")
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            with args.output.open("x") as destination:
                destination.write(encoded)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Gemma quantization plan rejected: {error}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a compact Laguna checkpoint with prepacked gate/up expert rows.

This is an offline packaging utility. The resulting checkpoint is consumed by
the native Swift/MLX runtime and does not require Python for inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as error:
    raise SystemExit(
        "pack-laguna-gate-up.py requires torch and safetensors in its Python environment"
    ) from error


INDEX_NAME = "model.safetensors.index.json"
GATE_WEIGHT_PATTERN = re.compile(
    r"^(language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp)\.gate_proj\.weight$"
)
PROJECTION_SUFFIXES = ("weight", "scales", "biases")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Laguna MLX checkpoint whose sparse expert gate/up projections "
            "are prepacked for the native fused Swift runtime."
        )
    )
    parser.add_argument("source", type=Path, help="Source MLX checkpoint directory")
    parser.add_argument(
        "destination",
        type=Path,
        help="New destination directory; it must not already exist",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_auxiliary_files(
    source: Path,
    staging: Path,
    indexed_weight_files: set[str],
) -> None:
    staging.mkdir()
    for item in sorted(source.iterdir()):
        if not item.is_file() or item.name == INDEX_NAME:
            continue
        if item.name in indexed_weight_files:
            continue
        destination = staging / item.name
        if item.suffix == ".safetensors":
            os.link(item, destination)
        else:
            shutil.copy2(item, destination)


def sparse_layers(weight_map: dict[str, str]) -> list[tuple[int, str]]:
    layers: list[tuple[int, str]] = []
    for key in weight_map:
        match = GATE_WEIGHT_PATTERN.match(key)
        if match:
            layers.append((int(match.group(2)), match.group(1)))
    layers.sort()
    if not layers:
        raise ValueError("No Laguna switch_mlp gate projections were found")
    return layers


def require_projection_pairs(weight_map: dict[str, str], prefix: str) -> None:
    missing: list[str] = []
    for suffix in PROJECTION_SUFFIXES:
        for projection in ("gate_proj", "up_proj"):
            key = f"{prefix}.{projection}.{suffix}"
            if key not in weight_map:
                missing.append(key)
    if missing:
        raise ValueError("Checkpoint is missing projection tensors: " + ", ".join(missing))


def projection_plan(
    weight_map: dict[str, str],
) -> tuple[list[tuple[int, str]], dict[str, list[tuple[int, str]]]]:
    layers = sparse_layers(weight_map)
    by_file: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for layer, prefix in layers:
        require_projection_pairs(weight_map, prefix)
        fused_keys = [f"{prefix}.gate_up_proj.{suffix}" for suffix in PROJECTION_SUFFIXES]
        if any(key in weight_map for key in fused_keys):
            raise ValueError(f"Layer {layer} already contains fused gate/up tensors")

        files = {
            weight_map[f"{prefix}.{projection}.{suffix}"]
            for projection in ("gate_proj", "up_proj")
            for suffix in PROJECTION_SUFFIXES
        }
        if len(files) != 1:
            raise ValueError(
                f"Layer {layer} gate/up tensors span multiple shards: {sorted(files)}"
            )
        by_file[next(iter(files))].append((layer, prefix))
    return layers, dict(by_file)


def build_compact_shards(
    source: Path,
    staging: Path,
    index: dict[str, object],
) -> tuple[list[int], int, dict[str, list[int]], list[str]]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError(f"{INDEX_NAME} has an invalid weight_map")

    layers, layers_by_file = projection_plan(weight_map)
    source_files = sorted(set(weight_map.values()))
    for filename in source_files:
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Index references a missing shard: {path}")

    fused_bytes = 0
    replaced_bytes = 0
    fused_shapes: dict[str, list[int]] = {}
    rewritten_files: list[str] = []

    for filename in source_files:
        source_path = source / filename
        output = staging / filename
        shard_layers = layers_by_file.get(filename, [])
        if not shard_layers:
            os.link(source_path, output)
            continue

        with safe_open(source_path, framework="pt", device="cpu") as handle:
            source_keys = set(handle.keys())
            removed_keys = {
                f"{prefix}.{projection}.{suffix}"
                for _, prefix in shard_layers
                for projection in ("gate_proj", "up_proj")
                for suffix in PROJECTION_SUFFIXES
            }
            packed: dict[str, torch.Tensor] = {
                key: handle.get_tensor(key)
                for key in handle.keys()
                if key not in removed_keys
            }

            for layer, prefix in shard_layers:
                for suffix in PROJECTION_SUFFIXES:
                    gate_key = f"{prefix}.gate_proj.{suffix}"
                    up_key = f"{prefix}.up_proj.{suffix}"
                    gate = handle.get_tensor(gate_key)
                    up = handle.get_tensor(up_key)
                    if gate.dtype != up.dtype:
                        raise ValueError(f"Dtype mismatch for layer {layer} {suffix}")
                    if (
                        gate.ndim < 2
                        or gate.shape[:-2] != up.shape[:-2]
                        or gate.shape[-1] != up.shape[-1]
                    ):
                        raise ValueError(f"Shape mismatch for layer {layer} {suffix}")

                    fused_key = f"{prefix}.gate_up_proj.{suffix}"
                    fused = torch.cat((gate, up), dim=-2).contiguous()
                    packed[fused_key] = fused
                    fused_shapes[fused_key] = list(fused.shape)
                    fused_bytes += fused.numel() * fused.element_size()
                    replaced_bytes += (
                        gate.numel() * gate.element_size()
                        + up.numel() * up.element_size()
                    )

                    del weight_map[gate_key]
                    del weight_map[up_key]
                    weight_map[fused_key] = filename

            metadata = dict(handle.metadata() or {})
            metadata["model_runner_optimization"] = "laguna-fused-gate-up-v2-compact"
            save_file(packed, output, metadata=metadata)
            expected_keys = (source_keys - removed_keys) | {
                f"{prefix}.gate_up_proj.{suffix}"
                for _, prefix in shard_layers
                for suffix in PROJECTION_SUFFIXES
            }

        shutil.copystat(source_path, output)
        with safe_open(output, framework="pt", device="cpu") as verification:
            if set(verification.keys()) != expected_keys:
                raise ValueError(f"Compact shard verification failed: {output}")

        rewritten_files.append(filename)
        layer_numbers = ",".join(str(layer) for layer, _ in shard_layers)
        print(f"repacked {filename} (layers {layer_numbers})", flush=True)
        del packed
        gc.collect()

    if fused_bytes != replaced_bytes:
        raise ValueError(
            f"Fused tensor bytes {fused_bytes} differ from replaced bytes {replaced_bytes}"
        )
    return [layer for layer, _ in layers], fused_bytes, fused_shapes, rewritten_files


def write_derived_metadata(
    source: Path,
    staging: Path,
    index: dict[str, object],
    layers: list[int],
    fused_bytes: int,
    fused_shapes: dict[str, list[int]],
    rewritten_files: list[str],
) -> None:
    index_path = staging / INDEX_NAME
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker = {
        "format": 2,
        "optimization": "laguna-fused-gate-up-v2-compact",
        "source": str(source),
        "source_index_sha256": sha256(source / INDEX_NAME),
        "fused_layer_count": len(layers),
        "fused_layers": layers,
        "fused_tensor_bytes": fused_bytes,
        "fused_shapes": fused_shapes,
        "rewritten_weight_files": rewritten_files,
    }
    (staging / "model-runner-laguna-fusion.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rewrite_fused_quantization_config(staging: Path, layers: list[int]) -> None:
    """Move matching gate/up precision overrides onto fused gate_up paths.

    A packed MLX tensor has one bit width and group size. Gate and up therefore
    must be one quantization unit when their rows are concatenated. Uniform Q4
    checkpoints need no override; Q4R8 checkpoints retain an explicit
    fused override when it differs from the global default.
    """
    config_path = staging / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint has no config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def effective(spec: dict[str, object], override: object) -> dict[str, object]:
        values = {
            "bits": spec.get("bits"),
            "group_size": spec.get("group_size"),
            "mode": spec.get("mode", "affine"),
        }
        if isinstance(override, dict):
            values.update(
                {key: override[key] for key in values if key in override}
            )
        return values

    changed = False
    for config_key in ("quantization", "quantization_config"):
        spec = config.get(config_key)
        if not isinstance(spec, dict):
            continue
        default = effective(spec, None)
        for layer in layers:
            prefixes = (
                f"language_model.model.layers.{layer}.mlp.switch_mlp",
                f"model.layers.{layer}.mlp.switch_mlp",
            )
            for prefix in prefixes:
                gate_key = f"{prefix}.gate_proj"
                up_key = f"{prefix}.up_proj"
                fused_key = f"{prefix}.gate_up_proj"
                gate = effective(spec, spec.get(gate_key))
                up = effective(spec, spec.get(up_key))
                if gate != up:
                    raise ValueError(
                        f"Layer {layer} gate/up quantization differs and cannot be fused: "
                        f"gate={gate}, up={up}"
                    )
                removed = spec.pop(gate_key, None) is not None
                removed = (spec.pop(up_key, None) is not None) or removed
                if gate != default and removed:
                    spec[fused_key] = gate
                    changed = True
                elif removed:
                    spec.pop(fused_key, None)
                    changed = True

    if changed:
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    arguments = parse_arguments()
    source = arguments.source.expanduser().resolve()
    destination = arguments.destination.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source checkpoint directory does not exist: {source}")
    if not (source / INDEX_NAME).is_file():
        raise SystemExit(f"Source checkpoint has no {INDEX_NAME}: {source}")
    if destination == source:
        raise SystemExit("Source and destination must differ")
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"Destination already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.packing-{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise SystemExit(f"Staging path already exists: {staging}")

    index = json.loads((source / INDEX_NAME).read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise SystemExit(f"{INDEX_NAME} must contain a JSON object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise SystemExit(f"{INDEX_NAME} has an invalid weight_map")
    indexed_weight_files = set(weight_map.values())

    try:
        copy_auxiliary_files(source, staging, indexed_weight_files)
        layers, fused_bytes, fused_shapes, rewritten_files = build_compact_shards(
            source, staging, index
        )
        rewrite_fused_quantization_config(staging, layers)
        write_derived_metadata(
            source,
            staging,
            index,
            layers,
            fused_bytes,
            fused_shapes,
            rewritten_files,
        )
        staging.rename(destination)
    except Exception:
        print(
            f"Packaging failed; partial output was preserved for inspection at {staging}",
            file=sys.stderr,
        )
        raise

    print(
        f"created compact checkpoint {destination} with {len(layers)} fused layers "
        f"({fused_bytes / (1024 ** 3):.2f} GiB replaced in-place)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

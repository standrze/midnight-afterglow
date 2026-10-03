#!/usr/bin/env python3
"""Verify that the Ollama Laguna compatibility view preserves tensor payloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import BinaryIO


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("q4r8_compat_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_regions(
    left: BinaryIO,
    left_offset: int,
    right: BinaryIO,
    right_offset: int,
    length: int,
) -> None:
    left.seek(left_offset)
    right.seek(right_offset)
    remaining = length
    while remaining:
        size = min(8 * 1024 * 1024, remaining)
        left_data = left.read(size)
        right_data = right.read(size)
        if left_data != right_data:
            raise ValueError(
                f"payload mismatch at left={left.tell() - len(left_data)}, "
                f"right={right.tell() - len(right_data)}"
            )
        remaining -= size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--compat-manifest", type=Path, required=True)
    parser.add_argument("--blob-dir", type=Path, required=True)
    args = parser.parse_args()

    builder = load_builder(args.builder)
    source = json.loads(args.source_manifest.read_text())
    compat = json.loads(args.compat_manifest.read_text())
    compat_layers = {
        layer["name"]: layer
        for layer in compat["layers"]
        if layer["mediaType"] == builder.TENSOR_MEDIA_TYPE
    }

    checked_layers = 0
    checked_payload_bytes = 0
    split_layers = 0
    q8_routers = 0

    for source_layer in source["layers"]:
        if source_layer["mediaType"] != builder.TENSOR_MEDIA_TYPE:
            continue
        source_name = source_layer["name"]
        normalized = builder.normalize_name(source_name)
        source_path = builder.digest_path(args.blob_dir, source_layer["digest"])
        source_header, source_data_start = builder.read_safetensors_header(source_path)

        if "gate_up_proj" not in normalized:
            destination_layer = compat_layers[normalized]
            destination_path = builder.digest_path(args.blob_dir, destination_layer["digest"])
            destination_header, destination_data_start = builder.read_safetensors_header(
                destination_path
            )
            source_bytes = source_path.stat().st_size - source_data_start
            destination_bytes = destination_path.stat().st_size - destination_data_start
            if source_bytes != destination_bytes:
                raise ValueError(f"data length changed for {source_name}")
            with source_path.open("rb") as left, destination_path.open("rb") as right:
                compare_regions(
                    left,
                    source_data_start,
                    right,
                    destination_data_start,
                    source_bytes,
                )
            if normalized.endswith(".mlp.gate.weight"):
                if destination_header.get("__metadata__", {}).get("quant_type") != "int8":
                    raise ValueError(f"router is not marked int8: {normalized}")
                q8_routers += 1
            checked_layers += 1
            checked_payload_bytes += source_bytes
            continue

        gate_name = normalized.replace("gate_up_proj", "gate_proj")
        up_name = normalized.replace("gate_up_proj", "up_proj")
        gate_layer = compat_layers[gate_name]
        up_layer = compat_layers[up_name]
        gate_path = builder.digest_path(args.blob_dir, gate_layer["digest"])
        up_path = builder.digest_path(args.blob_dir, up_layer["digest"])
        gate_header, gate_data_start = builder.read_safetensors_header(gate_path)
        up_header, up_data_start = builder.read_safetensors_header(up_path)

        with source_path.open("rb") as source_file, gate_path.open("rb") as gate_file, up_path.open(
            "rb"
        ) as up_file:
            for source_tensor_name, source_info in builder.tensor_entries(source_header):
                source_shape = source_info["shape"]
                axis = source_shape[-2]
                if axis % 2:
                    raise ValueError(f"odd fused axis for {source_tensor_name}")
                source_start, source_end = source_info["data_offsets"]
                outer = math.prod(source_shape[:-2])
                inner_bytes = (
                    math.prod(source_shape[-1:])
                    * builder.DTYPE_BYTES[source_info["dtype"]]
                )
                half_bytes = (axis // 2) * inner_bytes

                normalized_tensor = builder.normalize_name(source_tensor_name)
                gate_tensor_name = normalized_tensor.replace("gate_up_proj", "gate_proj")
                up_tensor_name = normalized_tensor.replace("gate_up_proj", "up_proj")
                gate_info = gate_header[gate_tensor_name]
                up_info = up_header[up_tensor_name]
                gate_start = gate_info["data_offsets"][0]
                up_start = up_info["data_offsets"][0]

                for outer_index in range(outer):
                    source_outer = source_data_start + source_start + outer_index * half_bytes * 2
                    compare_regions(
                        source_file,
                        source_outer,
                        gate_file,
                        gate_data_start + gate_start + outer_index * half_bytes,
                        half_bytes,
                    )
                    compare_regions(
                        source_file,
                        source_outer + half_bytes,
                        up_file,
                        up_data_start + up_start + outer_index * half_bytes,
                        half_bytes,
                    )
                checked_payload_bytes += source_end - source_start

        split_layers += 1
        checked_layers += 1

    if checked_layers != 599:
        raise ValueError(f"expected 599 source tensor layers, checked {checked_layers}")
    if split_layers != 79:
        raise ValueError(f"expected 79 split layers, checked {split_layers}")
    if q8_routers != 39:
        raise ValueError(f"expected 39 Q8 routers, checked {q8_routers}")

    print(
        json.dumps(
            {
                "source_tensor_layers_checked": checked_layers,
                "exact_gate_up_splits_checked": split_layers,
                "q8_routers_checked": q8_routers,
                "tensor_payload_bytes_checked": checked_payload_bytes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

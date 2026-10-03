#!/usr/bin/env python3
"""Build an Ollama-only Laguna namespace/layout view without requantizing tensors.

The experimental Ollama safetensors importer preserves MLX affine arrays but
Ollama 0.33.1 expects the stock Laguna namespace and separate gate/up tensors.
This script rewrites safetensors headers, splits already-fused gate/up rows at
their exact midpoint, and corrects the 39 router headers to affine INT8. Tensor
payload bytes are otherwise copied verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any, BinaryIO, Iterable


TENSOR_MEDIA_TYPE = "application/vnd.ollama.image.tensor"
JSON_MEDIA_TYPE = "application/vnd.ollama.image.json"

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--blob-dir", type=Path, required=True)
    parser.add_argument("--destination-manifest", type=Path, required=True)
    return parser.parse_args()


def digest_path(blob_dir: Path, digest: str) -> Path:
    algorithm, value = digest.split(":", 1)
    if algorithm != "sha256" or len(value) != 64:
        raise ValueError(f"unsupported digest {digest}")
    return blob_dir / f"sha256-{value}"


def normalize_name(name: str) -> str:
    if name.startswith("language_model.model."):
        name = "model." + name[len("language_model.model.") :]
    elif name.startswith("language_model.lm_head."):
        name = "lm_head." + name[len("language_model.lm_head.") :]
    elif name.startswith("language_model."):
        name = name[len("language_model.") :]

    name = name.replace(".mlp.gate.proj.", ".mlp.gate.")
    name = name.replace(
        ".mlp.gate.e_score_correction_bias",
        ".mlp.experts.e_score_correction_bias",
    )
    return name


def transform_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {normalize_name(key): transform_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [transform_json(item) for item in value]
    return value


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length))
    return header, 8 + header_length


def encoded_header(header: dict[str, Any]) -> bytes:
    payload = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    padding = (-len(payload)) % 8
    payload += b" " * padding
    return struct.pack("<Q", len(payload)) + payload


def tensor_entries(header: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries = [(name, info) for name, info in header.items() if name != "__metadata__"]
    entries.sort(key=lambda pair: pair[1]["data_offsets"][0])
    return entries


def validate_tensor_info(name: str, info: dict[str, Any]) -> int:
    dtype = info["dtype"]
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype {dtype} for {name}")
    elements = math.prod(info["shape"])
    expected = elements * DTYPE_BYTES[dtype]
    start, end = info["data_offsets"]
    if end - start != expected:
        raise ValueError(
            f"payload length mismatch for {name}: offsets={end - start}, expected={expected}"
        )
    return expected


class DigestWriter:
    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.hasher = hashlib.sha256()
        self.size = 0

    def write(self, data: bytes) -> None:
        self.handle.write(data)
        self.hasher.update(data)
        self.size += len(data)

    @property
    def digest(self) -> str:
        return self.hasher.hexdigest()


def copy_region(source: BinaryIO, writer: DigestWriter, offset: int, length: int) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        block = source.read(min(8 * 1024 * 1024, remaining))
        if not block:
            raise EOFError(f"unexpected EOF with {remaining} bytes remaining")
        writer.write(block)
        remaining -= len(block)


def commit_blob(temp_path: Path, writer: DigestWriter, blob_dir: Path) -> tuple[str, int]:
    digest = writer.digest
    destination = blob_dir / f"sha256-{digest}"
    if destination.exists():
        if destination.stat().st_size != writer.size:
            raise ValueError(f"digest collision with different size at {destination}")
        temp_path.unlink()
    else:
        os.replace(temp_path, destination)
    return f"sha256:{digest}", writer.size


def write_renamed_blob(
    source_path: Path,
    blob_dir: Path,
    manifest_name: str,
) -> tuple[str, int]:
    header, data_start = read_safetensors_header(source_path)
    renamed: dict[str, Any] = {}
    if "__metadata__" in header:
        renamed["__metadata__"] = dict(header["__metadata__"])
    for name, info in header.items():
        if name == "__metadata__":
            continue
        validate_tensor_info(name, info)
        renamed[normalize_name(name)] = info

    if manifest_name.endswith(".mlp.gate.weight"):
        metadata = renamed.setdefault("__metadata__", {})
        metadata["group_size"] = "64"
        metadata["quant_type"] = "int8"

    temp_handle = tempfile.NamedTemporaryFile(prefix="q4r8-compat-", dir=blob_dir, delete=False)
    temp_path = Path(temp_handle.name)
    try:
        writer = DigestWriter(temp_handle)
        writer.write(encoded_header(renamed))
        with source_path.open("rb") as source:
            copy_region(source, writer, data_start, source_path.stat().st_size - data_start)
        temp_handle.flush()
        os.fsync(temp_handle.fileno())
        temp_handle.close()
        return commit_blob(temp_path, writer, blob_dir)
    except BaseException:
        temp_handle.close()
        temp_path.unlink(missing_ok=True)
        raise


def split_info(
    entries: list[tuple[str, dict[str, Any]]],
    replacement: str,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], int, int, int]]]:
    output: dict[str, Any] = {}
    regions: list[tuple[dict[str, Any], int, int, int]] = []
    cursor = 0
    for name, info in entries:
        validate_tensor_info(name, info)
        shape = list(info["shape"])
        if len(shape) < 2 or shape[-2] % 2:
            raise ValueError(f"cannot split {name} with shape {shape} on axis -2")
        original_axis = shape[-2]
        shape[-2] //= 2
        output_name = normalize_name(name).replace("gate_up_proj", replacement)
        output_bytes = math.prod(shape) * DTYPE_BYTES[info["dtype"]]
        output[output_name] = {
            "dtype": info["dtype"],
            "shape": shape,
            "data_offsets": [cursor, cursor + output_bytes],
        }
        cursor += output_bytes

        outer = math.prod(info["shape"][:-2])
        inner_bytes = math.prod(info["shape"][-1:]) * DTYPE_BYTES[info["dtype"]]
        half_chunk_bytes = (original_axis // 2) * inner_bytes
        regions.append((info, outer, half_chunk_bytes, output_bytes))
    return output, regions


def write_split_blob(
    source_path: Path,
    blob_dir: Path,
    replacement: str,
) -> tuple[str, int]:
    source_header, data_start = read_safetensors_header(source_path)
    entries = tensor_entries(source_header)
    output_header, regions = split_info(entries, replacement)
    if "__metadata__" in source_header:
        output_header = {"__metadata__": dict(source_header["__metadata__"]), **output_header}

    temp_handle = tempfile.NamedTemporaryFile(prefix="q4r8-split-", dir=blob_dir, delete=False)
    temp_path = Path(temp_handle.name)
    try:
        writer = DigestWriter(temp_handle)
        writer.write(encoded_header(output_header))
        with source_path.open("rb") as source:
            for (name, info), (_, outer, half_chunk_bytes, output_bytes) in zip(entries, regions):
                start, _ = info["data_offsets"]
                side_offset = 0 if replacement == "gate_proj" else half_chunk_bytes
                written = 0
                for outer_index in range(outer):
                    region_start = (
                        data_start
                        + start
                        + outer_index * half_chunk_bytes * 2
                        + side_offset
                    )
                    copy_region(source, writer, region_start, half_chunk_bytes)
                    written += half_chunk_bytes
                if written != output_bytes:
                    raise ValueError(
                        f"split payload mismatch for {name}: wrote={written}, expected={output_bytes}"
                    )
        temp_handle.flush()
        os.fsync(temp_handle.fileno())
        temp_handle.close()
        return commit_blob(temp_path, writer, blob_dir)
    except BaseException:
        temp_handle.close()
        temp_path.unlink(missing_ok=True)
        raise


def write_json_blob(source_path: Path, blob_dir: Path) -> tuple[str, int]:
    data = json.loads(source_path.read_text())
    transformed = transform_json(data)
    payload = json.dumps(transformed, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    destination = blob_dir / f"sha256-{digest}"
    if not destination.exists():
        temp_handle = tempfile.NamedTemporaryFile(prefix="q4r8-json-", dir=blob_dir, delete=False)
        temp_path = Path(temp_handle.name)
        try:
            temp_handle.write(payload)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
            temp_handle.close()
            os.replace(temp_path, destination)
        except BaseException:
            temp_handle.close()
            temp_path.unlink(missing_ok=True)
            raise
    return f"sha256:{digest}", len(payload)


def transformed_layer(layer: dict[str, Any], digest: str, size: int, name: str) -> dict[str, Any]:
    result = dict(layer)
    result["digest"] = digest
    result["size"] = size
    result["name"] = name
    return result


def main() -> None:
    args = parse_args()
    if args.destination_manifest.exists():
        raise FileExistsError(f"refusing existing destination {args.destination_manifest}")
    args.blob_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(args.source_manifest.read_text())
    output_layers: list[dict[str, Any]] = []
    tensor_count = 0
    split_count = 0
    q8_router_count = 0

    for index, layer in enumerate(source_manifest["layers"], start=1):
        source_path = digest_path(args.blob_dir, layer["digest"])
        media_type = layer["mediaType"]
        name = layer.get("name", "")
        if media_type == TENSOR_MEDIA_TYPE:
            normalized = normalize_name(name)
            if "gate_up_proj" in normalized:
                for replacement in ("gate_proj", "up_proj"):
                    digest, size = write_split_blob(source_path, args.blob_dir, replacement)
                    output_name = normalized.replace("gate_up_proj", replacement)
                    output_layers.append(
                        transformed_layer(layer, digest, size, output_name)
                    )
                split_count += 1
                tensor_count += 2
            else:
                digest, size = write_renamed_blob(source_path, args.blob_dir, normalized)
                output_layers.append(transformed_layer(layer, digest, size, normalized))
                tensor_count += 1
                if normalized.endswith(".mlp.gate.weight"):
                    q8_router_count += 1
        elif media_type == JSON_MEDIA_TYPE:
            digest, size = write_json_blob(source_path, args.blob_dir)
            output_layers.append(transformed_layer(layer, digest, size, name))
        else:
            output_layers.append(layer)

        if index % 50 == 0 or index == len(source_manifest["layers"]):
            print(f"processed {index}/{len(source_manifest['layers'])} source layers", flush=True)

    if q8_router_count != 39:
        raise ValueError(f"expected 39 Q8 routers, found {q8_router_count}")
    if split_count != 79:
        raise ValueError(f"expected 79 fused gate/up layers, found {split_count}")

    output_manifest = dict(source_manifest)
    output_manifest["layers"] = output_layers
    payload = json.dumps(output_manifest, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    args.destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="manifest-", dir=args.destination_manifest.parent, delete=False
    ) as handle:
        temp_manifest = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_manifest, args.destination_manifest)
    print(
        f"wrote {args.destination_manifest}: {tensor_count} tensor layers, "
        f"{split_count} exact splits, {q8_router_count} Q8 routers",
        flush=True,
    )


if __name__ == "__main__":
    main()

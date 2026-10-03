#!/usr/bin/env python3
"""Read-only G64-to-G128 audit: header geometry, byte totals and every preserved tensor."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

SIZES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2, 'F16': 2,
         'BF16': 2, 'U32': 4, 'I32': 4, 'F32': 4, 'U64': 8, 'I64': 8, 'F64': 8}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def headers(root):
    root = Path(root)
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    mapping, result = index['weight_map'], {}
    for shard in sorted(set(mapping.values())):
        if Path(shard).name != shard:
            raise ValueError('Expected shard filenames without directory traversal')
        path = root / shard
        with path.open('rb') as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError('Truncated safetensors header')
            length, = struct.unpack('<Q', raw)
            if length > 64 * 1024 * 1024:
                raise ValueError('Oversized safetensors header')
            header = json.loads(stream.read(length))
        for key, item in header.items():
            if key == '__metadata__':
                continue
            if key in result or mapping.get(key) != shard:
                raise ValueError('Index/header coverage mismatch')
            shape = item['shape']
            begin, end = item['data_offsets']
            if (any(type(d) is not int or d < 0 for d in shape)
                or not 0 <= begin <= end <= path.stat().st_size - 8 - length
                or end - begin != math.prod(shape) * SIZES[item['dtype']]):
                raise ValueError('Invalid tensor geometry/range')
            result[key] = {'path': path, 'offset': 8 + length + begin,
                           'bytes': end - begin, 'shape': shape, 'dtype': item['dtype']}
    if set(mapping) != set(result):
        raise ValueError('Missing indexed tensors')
    return index, result


def tensor_sha(item):
    digest = hashlib.sha256()
    with item['path'].open('rb') as stream:
        stream.seek(item['offset'])
        remaining = item['bytes']
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError('Truncated tensor payload')
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def verify(template, candidate):
    a_index, a = headers(template)
    b_index, b = headers(candidate)
    if a_index['weight_map'] != b_index['weight_map']:
        raise ValueError('Tensor names or shard placement changed')
    configs = [json.loads((Path(p) / 'config.json').read_text()) for p in (template, candidate)]
    original, output = [c.get('quantization', c.get('quantization_config')) for c in configs]
    # Native Laguna sanitation creates fused gate/up paths that are absent from
    # split checkpoints. They inherit these defaults, regardless of raw overrides.
    for policy, group, label in ((original, 64, 'Template'), (output, 128, 'Candidate')):
        if not isinstance(policy, dict):
            raise ValueError(label + ' global quantization policy is missing')
        actual = {key: policy.get(key, 'affine' if key == 'mode' else None)
                  for key in ('bits', 'group_size', 'mode')}
        if actual != {'bits': 4, 'group_size': group, 'mode': 'affine'}:
            raise ValueError(label + ' global quantization policy must be affine Q4/G' + str(group))
    architecture = [{k: v for k, v in c.items() if k not in ('quantization', 'quantization_config')} for c in configs]
    if architecture[0] != architecture[1]:
        raise ValueError('Non-quantization model configuration changed')
    if configs[1].get('quantization_config') != output:
        raise ValueError('Candidate quantization aliases differ')
    modules = {key[:-7] for key in a if key.endswith('.scales')}
    preserved, changed = [], []
    for key, before in a.items():
        after = b[key]
        module, suffix = key.rsplit('.', 1)
        old = {k: original.get(module, {}).get(k, original.get(k, 'affine' if k == 'mode' else None)) for k in ('bits', 'group_size', 'mode')}
        new = {k: output.get(module, {}).get(k, output.get(k, 'affine' if k == 'mode' else None)) for k in ('bits', 'group_size', 'mode')}
        regroup = module in modules and old['bits'] == 4 and not module.endswith('.embed_tokens')
        expected = before['shape'].copy()
        if regroup:
            if old != {'bits': 4, 'group_size': 64, 'mode': 'affine'} or new != {'bits': 4, 'group_size': 128, 'mode': 'affine'}:
                raise ValueError('Unexpected projection policy: ' + module)
            if suffix in ('scales', 'biases'):
                if expected[-1] % 2:
                    raise ValueError('Odd G64 metadata dimension')
                expected[-1] //= 2
            elif suffix != 'weight':
                raise ValueError('Unexpected quantized tensor suffix')
        elif module in modules and new != old:
            raise ValueError('Preserved router/embedding policy changed: ' + module)
        if after['shape'] != expected or after['dtype'] != before['dtype']:
            raise ValueError('Unexpected tensor shape/dtype: ' + key)
        if regroup:
            changed.append(key)
        else:
            first, second = tensor_sha(before), tensor_sha(after)
            if first != second:
                raise ValueError('Preserved tensor differs: ' + key)
            preserved.append({'key': key, 'bytes': before['bytes'], 'sha256': first})
    totals = [sum(item['bytes'] for item in tensors.values()) for tensors in (a, b)]
    if b_index['metadata']['total_size'] != totals[1]:
        raise ValueError('Candidate index byte total differs from actual tensors')
    sidecars = []
    for name in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json', 'special_tokens_map.json', 'tokenizer.model', 'chat_template.jinja'):
        before, after = Path(template) / name, Path(candidate) / name
        if before.exists() != after.exists() or (before.exists() and sha(before) != sha(after)):
            raise ValueError('Tokenizer/generation sidecar changed: ' + name)
        if before.exists():
            sidecars.append({'name': name, 'sha256': sha(before)})
    return {'schema': 'laguna_group_size_audit_v1', 'status': 'passed',
            'template': str(Path(template).resolve()), 'candidate': str(Path(candidate).resolve()),
            'template_tensor_bytes': totals[0], 'candidate_tensor_bytes': totals[1],
            'saved_tensor_bytes': totals[0] - totals[1], 'saved_fraction': 1 - totals[1] / totals[0],
            'q4_tensor_headers_checked': len(changed), 'preserved_tensor_count': len(preserved),
            'preserved_tensors': preserved, 'sidecars': sidecars,
            'limitation': 'Preserved payloads are fully hashed. Regrouped Q4 tensors have geometry checks only; their quality is evaluated separately.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output exists; preserve earlier evidence')
    report = verify(args.template, args.candidate)
    report['auditor_sha256'] = sha(__file__)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Verified {report['preserved_tensor_count']} preserved tensors; saved {report['saved_tensor_bytes']} bytes")


if __name__ == '__main__':
    main()

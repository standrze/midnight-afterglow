import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('group_audit', Path(__file__).resolve().parents[2] / 'Scripts/verify-laguna-group-size.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def checkpoint(root, regroup):
    root.mkdir()
    modules = {'model.projection': 4, 'model.router': 8, 'model.embed_tokens': 4}
    config = {'model_type': 'laguna', 'quantization': {'bits': 4, 'group_size': 128 if regroup else 64,
              'mode': 'affine', 'model.router': {'bits': 8, 'group_size': 64, 'mode': 'affine'}}}
    if regroup:
        config['quantization']['model.embed_tokens'] = {'bits': 4, 'group_size': 64, 'mode': 'affine'}
        config['quantization_config'] = config['quantization']
    (root / 'config.json').write_text(json.dumps(config))
    header, payload = {}, bytearray()
    def add(key, shape, dtype):
        size = audit.SIZES[dtype]
        count = 1
        for value in shape:
            count *= value
        start = len(payload)
        payload.extend(b'\x01' * (count * size))
        header[key] = {'shape': shape, 'dtype': dtype, 'data_offsets': [start, len(payload)]}
    for name, bits in modules.items():
        add(name + '.weight', [1, 16 if bits == 4 else 32], 'U32')
        groups = 1 if regroup and name == 'model.projection' else 2
        for kind in ('scales', 'biases'):
            add(name + '.' + kind, [1, groups], 'BF16')
    add('model.norm.weight', [128], 'BF16')
    raw = json.dumps(header).encode()
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + payload)
    (root / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': len(payload)},
         'weight_map': {key: 'model.safetensors' for key in header}}))


class GroupSizeAuditTests(unittest.TestCase):
    def test_geometry_and_preserved_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            checkpoint(a, False); checkpoint(b, True)
            result = audit.verify(a, b)
            self.assertEqual(result['saved_tensor_bytes'], 4)
            self.assertEqual(result['preserved_tensor_count'], 7)
            self.assertEqual(result['q4_tensor_headers_checked'], 3)
            _, tensors = audit.headers(b)
            with (b / 'model.safetensors').open('r+b') as stream:
                stream.seek(tensors['model.router.weight']['offset'])
                stream.write(b'\x02')
            with self.assertRaisesRegex(ValueError, 'Preserved tensor differs'):
                audit.verify(a, b)

    def test_preserved_policy_and_byte_total(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            checkpoint(a, False); checkpoint(b, True)
            index = b / 'model.safetensors.index.json'
            data = json.loads(index.read_text()); data['metadata']['total_size'] += 1
            index.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'byte total'):
                audit.verify(a, b)
            data['metadata']['total_size'] -= 1; index.write_text(json.dumps(data))
            config = b / 'config.json'; data = json.loads(config.read_text())
            for name in ('quantization', 'quantization_config'):
                data[name]['model.embed_tokens']['group_size'] = 128
            config.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Preserved router/embedding policy'):
                audit.verify(a, b)

    def test_raw_overrides_cannot_hide_wrong_native_fused_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            checkpoint(a, False); checkpoint(b, True)
            config = b / 'config.json'; data = json.loads(config.read_text())
            for name in ('quantization', 'quantization_config'):
                # Every stored projection still resolves G128, but a synthesized
                # native gate_up module would inherit the invalid global G64.
                data[name]['model.projection'] = {'bits': 4, 'group_size': 128, 'mode': 'affine'}
                data[name]['group_size'] = 64
            config.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Candidate global quantization policy'):
                audit.verify(a, b)

            for name in ('quantization', 'quantization_config'):
                data[name]['group_size'] = 128
            config.write_text(json.dumps(data))
            config = a / 'config.json'; data = json.loads(config.read_text())
            for module in ('model.projection', 'model.embed_tokens'):
                data['quantization'][module] = {'bits': 4, 'group_size': 64, 'mode': 'affine'}
            data['quantization']['group_size'] = 128
            config.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Template global quantization policy'):
                audit.verify(a, b)

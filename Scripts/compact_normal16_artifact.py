"""AGQ4N001 research envelope; fixed codebook, exclusive atomic file publication."""
import math
import os
from pathlib import Path
import struct
import tempfile

NORMAL16_BITS = [49011, 48454, 48138, 47670, 47266, 46704, 45975, 44293, 11525, 13207, 13936, 14498, 14902, 15370, 15686, 16243]

def encode_normal16(payload):
    rows, columns = payload['rows'], payload['columns']
    if type(rows) is not int or type(columns) is not int or not (0 < rows <= 0xffffffff and 0 < columns <= 0xffffffff) or columns % 64:
        raise ValueError('Invalid AGQ4N001 geometry')
    if payload.get('codebookBits', NORMAL16_BITS) != NORMAL16_BITS:
        raise ValueError('AGQ4N001 identifies the fixed normal16 codebook')
    count = rows * columns
    specifications = [('codes', count // 2, 255), ('scaleBytes', count // 32, 255), ('offsets', count // 64, 65535), ('gains', rows, 65535)]
    for name, length, maximum in specifications:
        values = payload[name]
        if len(values) != length or any(type(v) is not int or not 0 <= v <= maximum for v in values):
            raise ValueError('Invalid AGQ4N001 ' + name)
    def bf16(bits):
        return struct.unpack('<f', struct.pack('<I', bits << 16))[0]
    offsets = [bf16(v) for v in payload['offsets']]
    gains = [bf16(v) for v in payload['gains']]
    if any(not math.isfinite(v) for v in offsets) or any(not math.isfinite(v) or v <= 0 for v in gains):
        raise ValueError('Invalid AGQ4N001 finite metadata')
    book = [struct.unpack('<e', struct.pack('<H', v))[0] for v in NORMAL16_BITS]
    maximum_float32 = float.fromhex('0x1.fffffep127')
    for index in range(count):
        code = (payload['codes'][index // 2] >> ((index % 2) * 4)) & 15
        scale = payload['scaleBytes'][index // 32]
        slope = (1 + (scale & 15) / 16) * 2 ** ((scale >> 4) - 7)
        normalized = book[code] * slope + offsets[index // 64]
        if abs(normalized) > maximum_float32 or abs(normalized * gains[index // columns]) > maximum_float32:
            raise ValueError('AGQ4N001 reconstructs non-finite FP32 weights')
    data = b'AGQ4N001' + struct.pack('<II', rows, columns)
    data += bytes(payload['codes']) + bytes(payload['scaleBytes'])
    data += struct.pack('<' + 'H' * len(payload['offsets']), *payload['offsets'])
    data += struct.pack('<' + 'H' * len(payload['gains']), *payload['gains'])
    checksum = 14695981039346656037
    for byte in data:
        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return data + struct.pack('<Q', checksum)

def write_normal16(payload, destination):
    destination = Path(destination)
    data = encode_normal16(payload)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        staged = Path(stream.name)
        stream.write(data)
    try:
        os.link(staged, destination)
    finally:
        staged.unlink()

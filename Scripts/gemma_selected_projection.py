"""Bounded raw BF16 Gemma projection row/expert reader for research probes.

Uses the native collector's source aliases and fused gate/up split convention.
Payload hashes cover selected bytes, not the whole checkpoint.
"""
import hashlib,json,math,re,struct,tempfile
from pathlib import Path

class SelectedProjectionReader:
    def __init__(self, source, maximum_payload_bytes=32*1024**2):
        self.source=Path(source).resolve()
        if type(maximum_payload_bytes) is not int or maximum_payload_bytes<=0:
            raise ValueError('positive payload budget required')
        self.maximum_payload_bytes=maximum_payload_bytes
        self.index=json.loads((self.source/'model.safetensors.index.json').read_text())['weight_map']

    def describe(self, path, expert=None):
        match=re.fullmatch(r'(?:language_model\.model|model)\.layers\.(\d+)\.(.+)',path)
        if not match: raise ValueError('unsupported native projection path')
        layer,relative=match.groups();half=None
        if relative in ('experts.switch_glu.gate_proj','experts.switch_glu.up_proj'):
            suffix='experts.gate_up_proj';half=0 if relative.endswith('gate_proj') else 1
        elif relative=='experts.switch_glu.down_proj':suffix='experts.down_proj'
        else:suffix=relative+'.weight'
        candidates=[f'{root}.layers.{layer}.{suffix}' for root in
                    ['model.language_model','language_model.model','language_model','model']]
        keys=[k for k in candidates if k in self.index]
        if len(keys)!=1:raise ValueError('missing or ambiguous source projection')
        key=keys[0];name=self.index[key]
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('invalid shard path')
        shard=(self.source/name).resolve()
        cache_blob_root=self.source.parent.parent/'blobs'
        valid_cache_blob=(self.source.parent.name=='snapshots'
                          and re.fullmatch(r'[0-9a-f]{40}',self.source.name) is not None
                          and shard.parent==cache_blob_root.resolve()
                          and re.fullmatch(r'[0-9a-f]{64}',shard.name) is not None)
        if not shard.is_relative_to(self.source) and not valid_cache_blob:
            raise ValueError('shard outside source or its same-repository HF blob store')
        with shard.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0]
            if not 0<n<=64*1024**2 or n>shard.stat().st_size-8:raise ValueError('invalid header length')
            info=json.loads(f.read(n))[key]
        shape=info['shape'];start,end=info['data_offsets']
        if info['dtype']!='BF16' or any(type(d) is not int or d<=0 for d in shape):raise ValueError('require positive BF16 geometry')
        if any(type(x) is not int for x in [start,end]) or start<0 or end<start or 8+n+end>shard.stat().st_size or end-start!=math.prod(shape)*2:raise ValueError('invalid payload offsets')
        routed=relative.startswith('experts.')
        if routed:
            if len(shape)!=3 or type(expert) is not int or not 0<=expert<shape[0]:raise ValueError('valid selected expert required')
            rows,width=shape[1:];offset=expert*rows*width*2
            if half is not None:
                if rows%2:raise ValueError('invalid fused gate/up geometry')
                rows//=2;offset+=half*rows*width*2
        else:
            if len(shape)!=2 or expert is not None:raise ValueError('dense projection must be a matrix without expert')
            rows,width=shape;offset=0
        return {'key':key,'shard':str(shard),'shape':shape,'projection_shape':[rows,width],
                'payload_start':8+n+start+offset,'dtype':'BF16','expert':expert,'fused_half':half}

    def read_bytes(self,path,rows,expert=None):
        info=self.describe(path,expert);count,width=info['projection_shape']
        if not rows or len(set(rows))!=len(rows) or any(type(r) is not int or not 0<=r<count for r in rows):raise ValueError('unique in-range rows required')
        if len(rows)*width*2>self.maximum_payload_bytes:raise MemoryError('selected payload exceeds budget')
        payload=bytearray()
        with Path(info['shard']).open('rb') as f:
            for row in rows:
                f.seek(info['payload_start']+row*width*2);chunk=f.read(width*2)
                if len(chunk)!=width*2:raise ValueError('truncated selected row')
                payload.extend(chunk)
        evidence={**info,'rows':list(rows),'selected_shape':[len(rows),width],
                  'selected_payload_sha256':hashlib.sha256(payload).hexdigest(),
                  'whole_checkpoint_identity_verified_by_this_reader':False}
        return bytes(payload),evidence

    def read(self,path,rows,expert=None):
        import mlx.core as mx
        payload,evidence=self.read_bytes(path,rows,expert)
        header=json.dumps({'selected':{'dtype':'BF16','shape':evidence['selected_shape'],
                                      'data_offsets':[0,len(payload)]}}).encode()
        header+=b' '*((-len(header))%8)
        with tempfile.NamedTemporaryFile(suffix='.safetensors') as f:
            f.write(struct.pack('<Q',len(header))+header+payload);f.flush()
            array=mx.load(f.name)['selected'];mx.eval(array)
        return array,evidence

"""New-output checkpoint transaction for fixed-shape native tensor patches.

Clones regular flat template files without hardlinks, writes bounded matrix rows,
and verifies that every byte outside registered patches is identical. No overwrite.
"""
import hashlib,json,math,os,shutil,struct,uuid
from pathlib import Path

SIZES={'BF16':2,'F16':2,'F32':4,'U32':4,'I32':4,'U64':8,'I64':8,'U8':1,'I8':1,'BOOL':1}

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()

def tensor_info(path,key):
    path=Path(path)
    with path.open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise ValueError('truncated safetensors prefix')
        n=struct.unpack('<Q',raw)[0]
        if not 0<n<=64*1024**2 or n>path.stat().st_size-8:raise ValueError('invalid safetensors header size')
        def unique(items):
            d={}
            for k,v in items:
                if k in d:raise ValueError('duplicate safetensors key')
                d[k]=v
            return d
        h=json.loads(f.read(n),object_pairs_hook=unique)
    info=h[key];shape=info['shape'];start,end=info['data_offsets'];dtype=info['dtype']
    if dtype not in SIZES or not shape or any(type(x) is not int or x<=0 for x in shape):raise ValueError('invalid tensor geometry/dtype')
    if any(type(x) is not int for x in [start,end]) or start<0 or end<start or end-start!=math.prod(shape)*SIZES[dtype] or 8+n+end>path.stat().st_size:raise ValueError('invalid tensor payload offsets')
    return info,8+n+start

def read_tensor_bytes(path,key,maximum_bytes=32*1024**2):
    info,start=tensor_info(path,key);n=math.prod(info['shape'])*SIZES[info['dtype']]
    if n>maximum_bytes:raise MemoryError('bounded tensor payload exceeded')
    with Path(path).open('rb') as f:f.seek(start);data=f.read(n)
    if len(data)!=n:raise ValueError('truncated tensor payload')
    return data,info

class CheckpointPatchTransaction:
    def __init__(self,template,destination,minimum_free_bytes):
        self.template=Path(template).resolve();requested=Path(destination)
        self.destination=requested.parent.resolve()/requested.name;self.minimum_free_bytes=minimum_free_bytes
        if type(minimum_free_bytes) is not int or minimum_free_bytes<0:raise ValueError('invalid disk reserve')
        if not self.template.is_dir() or not self.destination.parent.is_dir() or self.destination.exists() or self.destination.is_relative_to(self.template):raise ValueError('require a new output outside the existing template')
        entries=list(self.template.iterdir())
        if not entries or any(not p.is_file() or p.is_symlink() for p in entries):raise ValueError('research transaction requires regular flat template files')
        self.files=sorted(entries);self.stage=self.destination.parent/('.gemma-covariance-'+str(uuid.uuid4()));self.patches={};self.template_sha256={};self.committed=False
    def require_free(self,extra=0):
        if shutil.disk_usage(self.destination.parent).free<self.minimum_free_bytes+extra:raise OSError('checkpoint disk reserve would be crossed')
    def __enter__(self):
        self.require_free(sum(p.stat().st_size for p in self.files)+1024**3)
        self.stage.mkdir(mode=0o700)
        try:
            for source in self.files:
                digest=hashlib.sha256();dest=self.stage/source.name
                with source.open('rb') as inp,dest.open('xb') as out:
                    while data:=inp.read(8*1024**2):
                        self.require_free(len(data));out.write(data);digest.update(data)
                    out.flush();os.fsync(out.fileno())
                dest.chmod(0o600);self.template_sha256[source.name]=digest.hexdigest()
            return self
        except BaseException:
            shutil.rmtree(self.stage);raise
    def write_rows(self,shard,key,payload,*,dtype,row_start,row_count,expert=None):
        if Path(shard).name!=shard or shard not in self.template_sha256:raise ValueError('unknown template shard')
        info,base=tensor_info(self.stage/shard,key);shape=info['shape']
        if dtype!=info['dtype'] or type(row_start) is not int or type(row_count) is not int or row_start<0 or row_count<=0:raise ValueError('invalid patch dtype/row range')
        if len(shape)==2:
            if expert is not None:raise ValueError('dense patch cannot select an expert')
            rows,width=shape;start=base
        elif len(shape)==3:
            if type(expert) is not int or not 0<=expert<shape[0]:raise ValueError('valid expert required for stacked tensor')
            rows,width=shape[1:];start=base+expert*rows*width*SIZES[dtype]
        else:raise ValueError('patch requires matrix or stacked matrix')
        size=width*SIZES[dtype]
        if row_start+row_count>rows or len(payload)!=row_count*size:raise ValueError('patch row bounds or byte size mismatch')
        start+=row_start*size;end=start+len(payload)
        existing=self.patches.setdefault(shard,[])
        if any(start<p['end'] and p['start']<end for p in existing):raise ValueError('overlapping registered patches')
        with (self.stage/shard).open('r+b') as f:f.seek(start);f.write(payload)
        existing.append({'key':key,'expert':expert,'row_start':row_start,'row_count':row_count,'start':start,'end':end,'sha256':hashlib.sha256(payload).hexdigest()})
    def verify(self):
        proof={}
        for original in self.files:
            name=original.name;candidate=self.stage/name
            if sha_file(original)!=self.template_sha256[name]:raise ValueError('template changed during conversion')
            if candidate.stat().st_size!=original.stat().st_size:raise ValueError('candidate shape/byte size changed')
            ranges=sorted(self.patches.get(name,[]),key=lambda p:p['start']);offset=0;unchanged=hashlib.sha256();changed=[]
            with original.open('rb') as a,candidate.open('rb') as b:
                for item in ranges+[{'start':original.stat().st_size,'end':original.stat().st_size,'sha256':None}]:
                    remaining=item['start']-offset
                    if remaining<0:raise ValueError('invalid patch order')
                    while remaining:
                        n=min(8*1024**2,remaining);x=a.read(n);y=b.read(n)
                        if len(x)!=n or x!=y:raise ValueError('unregistered checkpoint bytes changed')
                        unchanged.update(x);remaining-=n
                    n=item['end']-item['start']
                    if n:
                        a.seek(n,1);data=b.read(n)
                        if len(data)!=n or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('patch bytes differ from fitted payload')
                        changed.append(item)
                    offset=item['end']
            proof[name]={'original_sha256':self.template_sha256[name],'candidate_sha256':sha_file(candidate),'unchanged_region_sha256':unchanged.hexdigest(),'patches':changed}
        return proof
    def publish(self,provenance,model_card):
        self.require_free(1024**2);proof=self.verify();provenance={**provenance,'template_byte_verification_before_metadata_edits':proof}
        for name,value in [('covariance-quantization.json',provenance),('model-card.json',model_card)]:
            data=(json.dumps(value,indent=2)+'\n').encode();self.require_free(len(data));(self.stage/name).write_bytes(data)
        if self.destination.exists():raise ValueError('output appeared during conversion')
        self.stage.rename(self.destination);self.committed=True;return proof
    def __exit__(self,*args):
        if not self.committed and self.stage.exists():shutil.rmtree(self.stage)

def read_matrix_rows(path,key,row_start,row_count,expert=None,maximum_bytes=32*1024**2):
    info,base=tensor_info(path,key);shape=info['shape'];size=SIZES[info['dtype']]
    if type(row_start) is not int or type(row_count) is not int or row_start<0 or row_count<=0:raise ValueError('invalid row range')
    if len(shape)==2:
        if expert is not None:raise ValueError('dense read cannot select expert')
        rows,width=shape
    elif len(shape)==3:
        if type(expert) is not int or not 0<=expert<shape[0]:raise ValueError('valid expert required')
        rows,width=shape[1:];base+=expert*rows*width*size
    else:raise ValueError('matrix geometry required')
    n=row_count*width*size
    if row_start+row_count>rows:raise ValueError('rows exceed tensor')
    if n>maximum_bytes:raise MemoryError('matrix row read exceeds byte budget')
    with Path(path).open('rb') as f:f.seek(base+row_start*width*size);data=f.read(n)
    if len(data)!=n:raise ValueError('truncated row read')
    return data,{'dtype':info['dtype'],'shape':[row_count,width],'data_offsets':[0,n]}

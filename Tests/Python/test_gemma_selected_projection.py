import json,struct,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Scripts'))
from gemma_selected_projection import SelectedProjectionReader

class SelectedProjectionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  arrays={'model.language_model.layers.0.experts.gate_up_proj':([3,8,4],list(range(96))),
          'model.language_model.layers.0.experts.down_proj':([3,4,2],list(range(100,124))),
          'model.language_model.layers.0.self_attn.q_proj.weight':([8,4],list(range(200,232)))}
  h={};payload=bytearray()
  for k,(shape,values) in arrays.items():
   start=len(payload);payload.extend(struct.pack('<'+'H'*len(values),*values));h[k]={'dtype':'BF16','shape':shape,'data_offsets':[start,len(payload)]}
  header=json.dumps(h).encode();(self.root/'weights.safetensors').write_bytes(struct.pack('<Q',len(header))+header+payload)
  (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:'weights.safetensors' for k in arrays}}))
  self.reader=SelectedProjectionReader(self.root)
 def tearDown(self):self.tmp.cleanup()
 def values(self,path,rows,expert=None):
  raw,e=self.reader.read_bytes(path,rows,expert);return struct.unpack('<'+'H'*(len(raw)//2),raw),e
 def test_fused_expert_halves_and_down(self):
  base='language_model.model.layers.0.experts.switch_glu.'
  gate,e=self.values(base+'gate_proj',[0,3],1);self.assertEqual(gate,(32,33,34,35,44,45,46,47));self.assertEqual(e['projection_shape'],[4,4])
  up,_=self.values(base+'up_proj',[0,3],1);self.assertEqual(up,(48,49,50,51,60,61,62,63))
  down,_=self.values(base+'down_proj',[3,0],2);self.assertEqual(down,(122,123,116,117))
 def test_dense_rows_and_payload_bound(self):
  path='language_model.model.layers.0.self_attn.q_proj';values,e=self.values(path,[7,0]);self.assertEqual(values,(228,229,230,231,200,201,202,203));self.assertFalse(e['whole_checkpoint_identity_verified_by_this_reader'])
  with self.assertRaises(MemoryError):SelectedProjectionReader(self.root,15).read_bytes(path,[0,1])
 def test_invalid_selection_and_aliases(self):
  path='language_model.model.layers.0.experts.switch_glu.gate_proj'
  for rows,expert in [([0],None),([0],3),([4],0),([0,0],0),([],0)]:
   with self.assertRaises(ValueError):self.reader.read_bytes(path,rows,expert)
  self.reader.index['model.layers.0.experts.gate_up_proj']='weights.safetensors'
  with self.assertRaises(ValueError):self.reader.describe(path,0)
 def test_same_repository_hf_snapshot_blob_link(self):
  repo=self.root/'models--maker--fixture';snapshot=repo/'snapshots'/('a'*40);snapshot.mkdir(parents=True);blobs=repo/'blobs';blobs.mkdir()
  blob=blobs/('b'*64);blob.write_bytes((self.root/'weights.safetensors').read_bytes());(snapshot/'weights.safetensors').symlink_to(blob)
  (snapshot/'model.safetensors.index.json').write_bytes((self.root/'model.safetensors.index.json').read_bytes())
  reader=SelectedProjectionReader(snapshot);payload,_=reader.read_bytes('language_model.model.layers.0.self_attn.q_proj',[0]);self.assertEqual(struct.unpack('<4H',payload),(200,201,202,203))
  (snapshot/'weights.safetensors').unlink();(snapshot/'weights.safetensors').symlink_to(self.root/'weights.safetensors')
  with self.assertRaises(ValueError):reader.describe('language_model.model.layers.0.self_attn.q_proj')
 def test_outside_shard_and_corrupt_offsets(self):
  path='language_model.model.layers.0.self_attn.q_proj';key='model.language_model.layers.0.self_attn.q_proj.weight';self.reader.index[key]='../weights.safetensors'
  with self.assertRaises(ValueError):self.reader.describe(path)
  self.reader.index[key]='weights.safetensors';data=(self.root/'weights.safetensors').read_bytes();n=struct.unpack('<Q',data[:8])[0];h=json.loads(data[8:8+n]);h[key]['data_offsets'][1]+=2;header=json.dumps(h).encode();(self.root/'weights.safetensors').write_bytes(struct.pack('<Q',len(header))+header+data[8+n:])
  with self.assertRaises(ValueError):self.reader.describe(path)
if __name__=='__main__':unittest.main()

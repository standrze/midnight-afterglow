import json,struct,tempfile,unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'Scripts'))
from gemma_checkpoint_patch import CheckpointPatchTransaction,read_tensor_bytes
class CheckpointPatchTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.template=self.root/'template';self.template.mkdir();self.dest=self.root/'candidate'
  h={'experts':{'shape':[3,4,2],'dtype':'U32','data_offsets':[0,96]},'dense':{'shape':[4,2],'dtype':'BF16','data_offsets':[96,112]}};header=json.dumps(h).encode();payload=struct.pack('<24I',*range(24))+struct.pack('<8H',*range(8));self.original=struct.pack('<Q',len(header))+header+payload;(self.template/'weights.safetensors').write_bytes(self.original);(self.template/'config.json').write_text('{"quantization":{"bits":4}}');(self.template/'model-card.json').write_text('{}')
 def tearDown(self):self.tmp.cleanup()
 def test_selected_expert_rows_preserve_all_other_bytes(self):
  with CheckpointPatchTransaction(self.template,self.dest,0) as t:
   t.write_rows('weights.safetensors','experts',struct.pack('<4I',90,91,92,93),dtype='U32',row_start=1,row_count=2,expert=1)
   t.write_rows('weights.safetensors','dense',struct.pack('<2H',50,51),dtype='BF16',row_start=3,row_count=1)
   proof=t.publish({'scope':'fixture'},{'name':'candidate'})
  values,_=read_tensor_bytes(self.dest/'weights.safetensors','experts');values=struct.unpack('<24I',values);expected=list(range(24));expected[10:14]=[90,91,92,93];self.assertEqual(list(values),expected);self.assertEqual((self.template/'weights.safetensors').read_bytes(),self.original);self.assertEqual((self.dest/'config.json').read_bytes(),(self.template/'config.json').read_bytes());self.assertEqual(len(proof['weights.safetensors']['patches']),2)
 def test_failure_rolls_back_and_existing_output_preserved(self):
  with self.assertRaises(RuntimeError):
   with CheckpointPatchTransaction(self.template,self.dest,0) as t:raise RuntimeError('injected fit failure')
  self.assertFalse(self.dest.exists());self.assertFalse(list(self.root.glob('.gemma-covariance-*')));self.dest.mkdir();(self.dest/'marker').write_text('keep')
  with self.assertRaises(ValueError):CheckpointPatchTransaction(self.template,self.dest,0)
  self.assertEqual((self.dest/'marker').read_text(),'keep')
 def test_invalid_expert_overlap_and_unregistered_mutation(self):
  with CheckpointPatchTransaction(self.template,self.dest,0) as t:
   payload=struct.pack('<2I',55,56)
   for expert in [None,3,-1]:
    with self.assertRaises(ValueError):t.write_rows('weights.safetensors','experts',payload,dtype='U32',row_start=0,row_count=1,expert=expert)
   t.write_rows('weights.safetensors','experts',payload,dtype='U32',row_start=0,row_count=1,expert=2)
   with self.assertRaises(ValueError):t.write_rows('weights.safetensors','experts',payload,dtype='U32',row_start=0,row_count=1,expert=2)
   with (t.stage/'weights.safetensors').open('r+b') as f:f.seek(-1,2);f.write(b'Z')
   with self.assertRaises(ValueError):t.publish({}, {})
  self.assertFalse(self.dest.exists())
 def test_changed_template_and_disk_reserve_abort(self):
  with CheckpointPatchTransaction(self.template,self.dest,0) as t:
   (self.template/'config.json').write_text('{"changed":true}')
   with self.assertRaises(ValueError):t.publish({}, {})
  with self.assertRaises(OSError):
   with CheckpointPatchTransaction(self.template,self.dest,10**30):pass
  self.assertFalse(self.dest.exists())
if __name__=='__main__':unittest.main()

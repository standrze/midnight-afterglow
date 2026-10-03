import hashlib,importlib.util,json,struct,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
SCRIPTS=Path(__file__).resolve().parents[2]/'Scripts';sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('converter',SCRIPTS/'convert-gemma-selective-covariance.py');converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
from gemma_checkpoint_patch import sha_file,tensor_info
from gemma_covariance_scale_search import inverse_factor,quantize

class SelectiveConverterTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import mlx.core as mx
  mx.set_default_device(mx.cpu);cls.mx=mx
 def setUp(self):
  mx=self.mx;self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.source=self.root/'repo/snapshots'/('a'*40);self.source.mkdir(parents=True);blobs=self.root/'repo/blobs';blobs.mkdir();self.template=self.root/'template';self.template.mkdir();self.dest=self.root/'candidate';self.fit=self.root/'fit';self.dev=self.root/'dev';self.fit.mkdir();self.dev.mkdir()
  dense=(mx.sin(mx.arange(129*128,dtype=mx.float32))*.1).reshape(129,128).astype(mx.bfloat16);fused=(mx.sin(mx.arange(3*6*128,dtype=mx.float32)*.21)*.2).reshape(3,6,128).astype(mx.bfloat16)
  self.native={'language_model.model.layers.0.self_attn.q_proj':dense,'language_model.model.layers.0.experts.switch_glu.gate_proj':fused[:,:3,:]}
  source_arrays={'model.language_model.layers.0.self_attn.q_proj.weight':dense,'model.language_model.layers.0.experts.gate_up_proj':fused};temp=self.root/'raw.safetensors';mx.save_safetensors(str(temp),source_arrays);blob=blobs/sha_file(temp);temp.rename(blob);(self.source/'weights.safetensors').symlink_to(blob)
  index={'weight_map':{k:'weights.safetensors' for k in source_arrays}};(self.source/'model.safetensors.index.json').write_text(json.dumps(index));(self.source/'config.json').write_text('{"fixture":true}');(self.source/'tokenizer.json').write_text('{}');(self.source/'tokenizer_config.json').write_text('{}')
  raw={};self.q_initial={}
  for path,array in self.native.items():
   q=mx.quantize(array,group_size=64,bits=4);self.q_initial[path]=q
   for suffix,value in zip(['weight','scales','biases'],q):raw[path+'.'+suffix]=value
  mx.save_safetensors(str(self.template/'weights.safetensors'),raw);tindex={k:'weights.safetensors' for k in raw};(self.template/'model.safetensors.index.json').write_text(json.dumps({'weight_map':tindex}));(self.template/'config.json').write_text('{"quantization":{"bits":4,"group_size":64,"mode":"affine"}}');(self.template/'model-card.json').write_text('{"name":"fixture"}');self.template_bytes={p.name:p.read_bytes() for p in self.template.iterdir()}
  identity={'directory':str(self.source),'configFingerprint':converter.fnv((self.source/'config.json').read_bytes()),'indexFingerprint':converter.fnv((self.source/'model.safetensors.index.json').read_bytes()),'tokenizerFingerprints':{n:converter.fnv((self.source/n).read_bytes()) for n in ['tokenizer.json','tokenizer_config.json']}}
  self.x=(mx.sin(mx.arange(256*128,dtype=mx.float32)*.17)).reshape(256,128).astype(mx.bfloat16);self.x[:,1:]+=self.x[:,:-1]*.3;mx.eval(self.x)
  paths=list(self.native);targets=[]
  for i,path in enumerate(paths):
   expert=None if i==0 else 1;shape=[129,128] if i==0 else [3,128];arrays={}
   for suffix in ['weight','scales','biases']:
    key=path+'.'+suffix;arrays[key]={'shard':'weights.safetensors','description':tensor_info(self.template/'weights.safetensors',key)[0]}
   targets.append({'path':path,'expert':expert,'source_shape':shape,'all_output_rows':shape[0],'input_fit':str(self.fit),'input_dev':str(self.dev),'template_arrays':arrays})
  for split,folder in [('fit',self.fit),('dev',self.dev)]:
   entries=[]
   for i,target in enumerate(targets):
    filename=f'input-{i}.safetensors';x=self.x if split=='fit' else self.x*.97;mx.save_safetensors(str(folder/filename),{'inputs':x});entries.append({'path':target['path'],'expert':target['expert'],'file':filename,'shape':[256,128],'capturedPositions':256})
   (folder/'manifest.json').write_text(json.dumps({'entries':entries,'provenance':{'source':identity,'sourceFamilies':[split+'-fixture']}}))
  self.plan=self.root/'plan.json';self.plan.write_text(json.dumps({'quantization':{'bits':4,'group_size':64,'metadata_dtype':'BF16','relative_damping':.01,'scale_factors':[1,.75,.8125,.875,.9375,1.0625,1.125,1.1875,1.25],'output_row_chunk':128,'refinement_steps':2},'minimum_free_bytes':0,'fitting_budget_bytes':128*1024**2,'candidates':[{'model':'a4b','source':str(self.source),'template':str(self.template),'output':str(self.dest),'template_config_sha256':sha_file(self.template/'config.json'),'template_index_sha256':sha_file(self.template/'model.safetensors.index.json'),'targets':targets}]}))
  self.args=types.SimpleNamespace(plan=str(self.plan),model='a4b',progress=str(self.root/'progress.json'))
 def tearDown(self):self.tmp.cleanup()
 def test_complete_conversion_native_arrays_and_untouched_experts(self):
  mx=self.mx;converter.convert(self.args);loaded=mx.load(str(self.dest/'weights.safetensors'));factor,_=inverse_factor(self.x)
  for path,array in self.native.items():
   expert=None if 'self_attn' in path else 1;weight=array if expert is None else array[expert];expected,_=quantize(weight,factor,64)
   actual=tuple(loaded[path+'.'+s] if expert is None else loaded[path+'.'+s][expert] for s in ['weight','scales','biases'])
   for a,b in zip(actual,expected):self.assertTrue(bool(mx.array_equal(a,b).item()))
   if expert is not None:
    for e in [0,2]:
     for suffix,original in zip(['weight','scales','biases'],self.q_initial[path]):self.assertTrue(bool(mx.array_equal(loaded[path+'.'+suffix][e],original[e]).item()))
   if expert is not None:
    stack=tuple(loaded[path+'.'+suffix] for suffix in ['weight','scales','biases'])
    gathered=mx.gather_qmm(self.x[None,:,:],*stack,lhs_indices=mx.array([0,0,0],dtype=mx.uint32),rhs_indices=mx.array([2,1,0],dtype=mx.uint32),transpose=True,group_size=64,bits=4)
    mx.eval(gathered)
    for i,e in enumerate([2,1,0]):
     reference=mx.quantized_matmul(self.x,*(v[e] for v in stack),transpose=True,group_size=64,bits=4)
     self.assertTrue(bool(mx.array_equal(gathered[i],reference).item()))
   output=mx.quantized_matmul(self.x,*actual,transpose=True,group_size=64,bits=4);mx.eval(output);self.assertTrue(bool(mx.all(mx.isfinite(output)).item()));self.assertEqual(output.shape,(256,weight.shape[0]))
  for name,data in self.template_bytes.items():self.assertEqual((self.template/name).read_bytes(),data)
  self.assertEqual((self.dest/'config.json').read_bytes(),self.template_bytes['config.json']);report=json.loads((self.dest/'covariance-quantization.json').read_text());self.assertEqual(len(report['targets']),2);self.assertEqual(report['recipe']['refinement_steps'],2);self.assertEqual(report['targets'][0]['chunks'][0]['fitting_detail']['algorithm'],'greedy-native-affine-q4-covariance-scale-search-v2');self.assertFalse(report['generated_quality_measured']);self.assertEqual(json.loads(Path(self.args.progress).read_text())['status'],'complete_checkpoint_unbenchmarked')
 def test_failed_fit_removes_staging_and_preserves_source_template(self):
  with patch('gemma_covariance_scale_search.quantize',side_effect=RuntimeError('injected failed fit')):
   with self.assertRaises(RuntimeError):converter.convert(self.args)
  self.assertFalse(self.dest.exists());self.assertFalse(list(self.root.glob('.gemma-covariance-*')))
  for name,data in self.template_bytes.items():self.assertEqual((self.template/name).read_bytes(),data)
if __name__=='__main__':unittest.main()

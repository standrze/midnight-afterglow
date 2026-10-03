import importlib.util,sys,unittest
from pathlib import Path
scripts=Path(__file__).resolve().parents[2]/'Scripts';sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('probe',scripts/'probe-gemma-covariance.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class ProjectionSearchTests(unittest.TestCase):
 def test_native_candidate_and_stored_grid_weighted_objective(self):
  import mlx.core as mx
  mx.set_default_device(mx.cpu)
  for dtype in [mx.bfloat16,mx.float16,mx.float32]:
   weight=(mx.sin(mx.arange(256,dtype=mx.float32))*.3).reshape(2,128).astype(dtype);importance=mx.linspace(.01,2,128)
   for group in [64,128]:
    native=mx.quantize(weight,group_size=group,bits=4);searched=probe.independent_search(mx,weight,importance,group)
    self.assertEqual(searched[1].dtype,dtype);self.assertEqual(sum(x.nbytes for x in searched),sum(x.nbytes for x in native))
    def loss(q):
     y=mx.dequantize(q[0],q[1].astype(mx.float32),q[2].astype(mx.float32),group_size=group,bits=4)
     return mx.sum(((weight.astype(mx.float32)-y)**2*importance).reshape(2,-1,group),axis=-1)
    self.assertTrue(bool(mx.all(loss(searched)<=loss(native)+1e-7).item()))
 def test_zero_and_constant_reconstruct_finitely(self):
  import mlx.core as mx
  mx.set_default_device(mx.cpu)
  for value in [0.,.25]:
   w=mx.full((2,128),value,dtype=mx.bfloat16);q=probe.independent_search(mx,w,mx.ones((128,)),64);out=mx.dequantize(*q,group_size=64,bits=4)
   self.assertTrue(bool(mx.all(mx.isfinite(out)).item()));self.assertTrue(bool(mx.all(out==w).item()))
if __name__=='__main__':unittest.main()

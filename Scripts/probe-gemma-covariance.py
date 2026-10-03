#!/usr/bin/env python3
"""Actual Gemma input reconstruction screen, not generated model quality."""
import argparse,hashlib,json,resource,time
from pathlib import Path
from gemma_selected_projection import SelectedProjectionReader
from gemma_covariance_resources import preflight

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent_search(mx,weight,importance,group):
    from gemma_covariance_scale_search import FACTORS
    rows,width=weight.shape;blocks=weight.reshape(rows,-1,group).astype(mx.float32)
    native=mx.quantize(weight,group_size=group,bits=4);s,b=native[1:];center=b.astype(mx.float32)+s.astype(mx.float32)*7.5
    shifts=mx.arange(0,32,4,dtype=mx.uint32)
    codes=((native[0][...,None]>>shifts)&15).reshape(rows,-1,group)
    best=codes.astype(mx.float32)*s[...,None].astype(mx.float32)+b[...,None].astype(mx.float32)
    loss=mx.sum((blocks-best)**2*importance.reshape(1,-1,group),axis=-1)
    best_s,best_b=s,b
    for f in FACTORS[1:]:
        cs=(s.astype(mx.float32)*f).astype(weight.dtype);cb=(center-cs.astype(mx.float32)*7.5).astype(weight.dtype)
        q=mx.where(cs[...,None]!=0,mx.clip(mx.round((blocks-cb[...,None].astype(mx.float32))/mx.where(cs[...,None]!=0,cs[...,None].astype(mx.float32),1)),0,15),0).astype(mx.uint32)
        reconstructed=q.astype(mx.float32)*cs[...,None].astype(mx.float32)+cb[...,None].astype(mx.float32)
        candidate=mx.sum((blocks-reconstructed)**2*importance.reshape(1,-1,group),axis=-1);take=candidate<loss
        codes=mx.where(take[...,None],q,codes);best_s=mx.where(take,cs,best_s);best_b=mx.where(take,cb,best_b);loss=mx.minimum(loss,candidate);mx.eval(codes,best_s,best_b,loss)
    shifts=mx.arange(0,32,4,dtype=mx.uint32);packed=mx.sum(codes.reshape(rows,-1,8)<<shifts,axis=-1).astype(mx.uint32)
    result=(packed,best_s,best_b);mx.eval(result);return result

def run(args):
    import mlx.core as mx
    from gemma_covariance_scale_search import inverse_factor,quantize
    mx.set_default_device(mx.cpu)
    output=Path(args.output)
    if output.exists():raise ValueError('existing report protected')
    fitdir,devdir=Path(args.fit),Path(args.dev)
    fm=json.loads((fitdir/'manifest.json').read_text());dm=json.loads((devdir/'manifest.json').read_text())
    if fm['provenance']['source']!=dm['provenance']['source']:raise ValueError('fit/development sources differ')
    if set(fm['provenance']['sourceFamilies'])&set(dm['provenance']['sourceFamilies']):raise ValueError('overlapping fit/dev source families')
    if Path(fm['provenance']['source']['directory']).resolve()!=Path(args.source).resolve():raise ValueError('source binding differs')
    def entry(m):
        found=[e for e in m['entries'] if e['path']==args.path and e.get('expert')==args.expert]
        if len(found)!=1:raise ValueError('missing or ambiguous actual input selection')
        return found[0]
    fe,de=entry(fm),entry(dm);fp,dp=fitdir/fe['file'],devdir/de['file']
    input_pins={str(p):sha(p) for p in [fitdir/'manifest.json',devdir/'manifest.json',fp,dp]}
    reader=SelectedProjectionReader(args.source);description=reader.describe(args.path,args.expert);count,width=description['projection_shape'];n=min(args.rows,count)
    rows=[i*(count-1)//(n-1) for i in range(n)] if n>1 else [0]
    plan=preflight(fe['capturedPositions'],width,n,args.budget_bytes)
    weight,evidence=reader.read(args.path,rows,args.expert);x=mx.load(str(fp))['inputs'];dev=mx.load(str(dp))['inputs'];mx.eval(weight,x,dev)
    if list(x.shape)!=fe['shape'] or list(dev.shape)!=de['shape'] or x.shape[1]!=width or dev.shape[1]!=width:raise ValueError('input geometry mismatch')
    if x.dtype!=mx.bfloat16 or dev.dtype!=mx.bfloat16 or not bool(mx.all(mx.isfinite(x)).item()) or not bool(mx.all(mx.isfinite(dev)).item()):raise ValueError('invalid actual BF16 inputs')
    started=time.monotonic();factor,coverage=inverse_factor(x,args.damping);methods=[]
    teacher={'fit':x.astype(mx.float32)@weight.astype(mx.float32).T,'dev':dev.astype(mx.float32)@weight.astype(mx.float32).T};mx.eval(teacher)
    for group in [64,128]:
        if width%group:continue
        for method in ['native_affine','unweighted_scale_search','diagonal_activation_scale_search','covariance_gptq','covariance_scale_search_v1','covariance_scale_search']:
            t=time.monotonic();detail={}
            if method=='native_affine':q=mx.quantize(weight,group_size=group,bits=4)
            elif method=='unweighted_scale_search':q=independent_search(mx,weight,mx.ones((width,),dtype=mx.float32),group)
            elif method=='diagonal_activation_scale_search':q=independent_search(mx,weight,mx.mean(x.astype(mx.float32)**2,axis=0),group)
            else:q,detail=quantize(weight,factor,group,factors=(1.0,) if method=='covariance_gptq' else (1.0,.75,.8125,.875,.9375,1.0625,1.125,1.1875,1.25),refinement_steps=0 if method in ('covariance_gptq','covariance_scale_search_v1') else 2)
            restored=mx.dequantize(*q,group_size=group,bits=4).astype(mx.float32);mx.eval(restored)
            scores={}
            for split,inputs in [('fit',x),('dev',dev)]:
                predicted=inputs.astype(mx.float32)@restored.T;error=predicted-teacher[split];denominator=float(mx.sum(teacher[split]**2).item());numerator=float(mx.sum(error**2).item())
                if denominator<=0 or not bool(mx.all(mx.isfinite(predicted)).item()):raise ValueError('invalid reconstruction measurement')
                scores[split]={'relative_output_mse':numerator/denominator,'positions':inputs.shape[0]}
            methods.append({'method':method,'group_size':group,'bits':4,'metadata_dtype':str(q[1].dtype),'selected_tensor_bytes':sum(v.nbytes for v in q),'seconds_research_cpu':time.monotonic()-t,'scores':scores,'fitting_detail':detail})
    if input_pins!={p:sha(p) for p in input_pins}:raise ValueError('input artifacts changed')
    _,after=reader.read_bytes(args.path,rows,args.expert)
    if evidence!=after:raise ValueError('selected source rows changed')
    report={'format':'gemma_actual_input_covariance_probe_v1','path':args.path,'expert':args.expert,'selected_source':evidence,'input_sha256':input_pins,'coverage':coverage,'planned_resources':plan.__dict__,'methods':methods,'unsupported_groups':[g for g in [64,128] if width%g],'seconds_total':time.monotonic()-started,'maximum_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'generated_quality_measured':False,'runtime_speed_measured':False,'limitations':[f'{n} deterministic output rows and at most 4096 prefix inputs; not full projection or model accuracy.','Fit/dev are family-disjoint development calibration corpora, not a fresh quality holdout.','Unweighted and diagonal search are source-matched native-grid controls, not reproductions of installed LS2/AWSS checkpoints.','Native source full-content FNV provenance comes from the verified capture; this reader checks selected source bytes.','Planning bytes are not a process footprint bound.']}
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--fit',required=True);p.add_argument('--dev',required=True);p.add_argument('--path',required=True);p.add_argument('--expert',type=int);p.add_argument('--output',required=True);p.add_argument('--rows',type=int,default=128);p.add_argument('--budget-bytes',type=int,default=8*1024**3);p.add_argument('--damping',type=float,default=.01);a=p.parse_args()
    if not 1<=a.rows<=128:p.error('rows must be 1...128')
    run(a)

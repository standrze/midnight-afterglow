#!/usr/bin/env python3
"""Research-only complete checkpoint with selective native-affine covariance Q4.

Untouched packed tensors retain the incumbent recipe. This does not promote a
model default, and conversion success does not establish generated quality.
"""
import argparse,hashlib,json,math,os,struct,time
from pathlib import Path
from gemma_checkpoint_patch import CheckpointPatchTransaction,read_matrix_rows,read_tensor_bytes,sha_file,tensor_info
from gemma_selected_projection import SelectedProjectionReader
from gemma_covariance_resources import preflight

def fnv(data):
    value=14695981039346656037
    for byte in data:value=((value^byte)*1099511628211)&0xffffffffffffffff
    return f'fnv1a64:{value:016x}'

def source_inventory(source,notify):
    source=Path(source);index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    names=sorted(set(index.values())|{'config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json'})
    result={}
    for name in names:
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('invalid source inventory path')
        path=source/name;notify('hashing_source',file=name);digest=sha_file(path)
        if name.endswith('.safetensors'):
            blob=path.resolve()
            if blob.parent.name!='blobs' or len(blob.name)!=64 or digest!=blob.name:raise ValueError('source weight differs from pinned content-addressed maker blob')
        result[name]={'sha256':digest,'bytes':path.stat().st_size}
    return result

def load_quantized_rows(mx,template,index,path,row,count,expert,temp):
    payloads=[];headers={};offset=0
    for suffix in ['weight','scales','biases']:
        key=path+'.'+suffix;data,desc=read_matrix_rows(template/index[key],key,row,count,expert)
        headers[suffix]={**desc,'data_offsets':[offset,offset+len(data)]};offset+=len(data);payloads.append(data)
    h=json.dumps(headers).encode();h+=b' '*((-len(h))%8)
    with temp.open('wb') as f:
        f.write(struct.pack('<Q',len(h))+h)
        for data in payloads:f.write(data)
    arrays=mx.load(str(temp));result=tuple(arrays[k] for k in ['weight','scales','biases']);mx.eval(result);temp.unlink();return result

def convert(args):
    import mlx.core as mx
    from gemma_covariance_scale_search import inverse_factor,quantize
    mx.set_default_device(mx.cpu)
    plan_path=Path(args.plan);plan=json.loads(plan_path.read_text());candidate=next(c for c in plan['candidates'] if c['model']==args.model);qplan=dict(plan['quantization']);qplan.setdefault('refinement_steps',0)
    if qplan['bits']!=4 or qplan['group_size']!=64 or qplan['metadata_dtype']!='BF16' or qplan['output_row_chunk']!=128:raise ValueError('unsupported preregistered recipe')
    source=Path(candidate['source']);template=Path(candidate['template']);destination=Path(candidate['output']);progress_path=Path(args.progress)
    if progress_path.exists():raise ValueError('existing conversion progress protected')
    progress={'status':'preflight','pid':os.getpid(),'model':args.model,'plan_sha256':sha_file(plan_path),'completed_targets':[],'generated_quality_measured':False}
    def notify(status,**fields):
        progress.update(status=status,**fields);temp=progress_path.with_suffix('.tmp');temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(progress_path)
    if sha_file(template/'config.json')!=candidate['template_config_sha256'] or sha_file(template/'model.safetensors.index.json')!=candidate['template_index_sha256']:raise ValueError('template metadata changed since planning')
    index=json.loads((template/'model.safetensors.index.json').read_text())['weight_map'];reader=SelectedProjectionReader(source);input_hashes={};provenances=[]
    for target in candidate['targets']:
        for split in ['fit','dev']:
            root=Path(target['input_'+split]);manifest=json.loads((root/'manifest.json').read_text());entry=[e for e in manifest['entries'] if e['path']==target['path'] and e.get('expert')==target['expert']]
            if len(entry)!=1 or manifest['provenance']['source']['directory']!=str(source):raise ValueError('capture binding mismatch')
            input_hashes[str(root/'manifest.json')]=sha_file(root/'manifest.json');input_hashes[str(root/entry[0]['file'])]=sha_file(root/entry[0]['file']);target[split+'_entry']=entry[0]
            if manifest['provenance'] not in provenances:provenances.append(manifest['provenance'])
        if reader.describe(target['path'],target['expert'])['projection_shape']!=target['source_shape']:raise ValueError('source geometry changed')
        for key,expected in target['template_arrays'].items():
            if index[key]!=expected['shard'] or tensor_info(template/index[key],key)[0]!=expected['description']:raise ValueError('template tensor geometry changed')
    if len(provenances)!=2 or set(provenances[0]['sourceFamilies'])&set(provenances[1]['sourceFamilies']):raise ValueError('require two family-disjoint capture splits')
    for provenance in provenances:
        identity=provenance['source']
        if fnv((source/'config.json').read_bytes())!=identity['configFingerprint'] or fnv((source/'model.safetensors.index.json').read_bytes())!=identity['indexFingerprint']:raise ValueError('source config/index differs from capture')
        for name,digest in identity['tokenizerFingerprints'].items():
            if fnv((source/name).read_bytes())!=digest:raise ValueError('source tokenizer differs from capture')
    before=source_inventory(source,notify)
    started=time.monotonic();target_reports=[]
    transaction=CheckpointPatchTransaction(template,destination,plan['minimum_free_bytes']+1024**3)
    notify('cloning_template',staging_directory=str(transaction.stage))
    with transaction:
        notify('fitting_full_output_rows',staging_directory=str(transaction.stage))
        for target in candidate['targets']:
            path,expert=target['path'],target['expert'];out,width=target['source_shape'];fit_path=Path(target['input_fit'])/target['fit_entry']['file'];dev_path=Path(target['input_dev'])/target['dev_entry']['file']
            resource_plan=preflight(target['fit_entry']['capturedPositions'],width,128,plan['fitting_budget_bytes']);x=mx.load(str(fit_path))['inputs'];dev=mx.load(str(dev_path))['inputs'];mx.eval(x,dev)
            if list(x.shape)!=target['fit_entry']['shape'] or list(dev.shape)!=target['dev_entry']['shape'] or x.dtype!=mx.bfloat16 or dev.dtype!=mx.bfloat16:raise ValueError('invalid captured input geometry/dtype')
            factor,coverage=inverse_factor(x,qplan['relative_damping']);mx.eval(factor)
            metrics={s:{'teacher_sum_squares':0.,'candidate_error_sum_squares':0.,'incumbent_error_sum_squares':0.} for s in ['fit','dev']};chunks=[];factor_counts=[0]*len(qplan['scale_factors'])
            for row in range(0,out,128):
                count=min(128,out-row);rows=list(range(row,row+count));weight,source_evidence=reader.read(path,rows,expert)
                q,detail=quantize(weight,factor,64,factors=tuple(qplan['scale_factors']),refinement_steps=qplan['refinement_steps']);incumbent=load_quantized_rows(mx,template,index,path,row,count,expert,transaction.stage/'.incumbent-chunk.safetensors')
                restored=mx.dequantize(*q,group_size=64,bits=4).astype(mx.float32);old=mx.dequantize(*incumbent,group_size=64,bits=4).astype(mx.float32);mx.eval(restored,old)
                for split,inputs in [('fit',x),('dev',dev)]:
                    teacher=inputs.astype(mx.float32)@weight.astype(mx.float32).T;pred=inputs.astype(mx.float32)@restored.T;baseline=inputs.astype(mx.float32)@old.T
                    vals=[float(mx.sum(v).item()) for v in [teacher**2,(pred-teacher)**2,(baseline-teacher)**2]]
                    if not all(math.isfinite(v) and v>=0 for v in vals):raise ValueError('nonfinite reconstruction')
                    for name,value in zip(metrics[split],vals):metrics[split][name]+=value
                temp=transaction.stage/'.quantized-chunk.safetensors';mx.save_safetensors(str(temp),dict(zip(['weight','scales','biases'],q)));patches=[]
                for suffix,dtype in [('weight','U32'),('scales','BF16'),('biases','BF16')]:
                    payload,info=read_tensor_bytes(temp,suffix);key=path+'.'+suffix;expected_width=width//8 if suffix=='weight' else width//64
                    if info['dtype']!=dtype or info['shape']!=[count,expected_width]:raise ValueError('fitted native array geometry/dtype mismatch')
                    transaction.write_rows(index[key],key,payload,dtype=dtype,row_start=row,row_count=count,expert=expert);patches.append({'key':key,'payload_sha256':hashlib.sha256(payload).hexdigest()})
                temp.unlink()
                _,source_after=reader.read_bytes(path,rows,expert)
                if source_evidence!=source_after:raise ValueError('source rows changed during fitting')
                for decision in detail['decisions']:
                    for i,n in enumerate(decision['factor_counts']):factor_counts[i]+=n
                chunks.append({'row_start':row,'row_count':count,'source':source_evidence,'patches':patches,'fitting_detail':detail,'conditional_error_scope':detail.get('decisions_scope','selected legacy pass'),'native_conditional_error_sum':sum(d['native_conditional_error'] for d in detail['decisions']),'selected_conditional_error_sum':sum(d['selected_conditional_error'] for d in detail['decisions'])})
                notify('fitting_full_output_rows',active_target=path,active_expert=expert,completed_rows=row+count,total_rows=out)
                del weight,q,incumbent,restored,old
            if sum(c['row_count'] for c in chunks)!=out:raise ValueError('incomplete full-row fit')
            for split,scores in metrics.items():
                if scores['teacher_sum_squares']<=0:raise ValueError('zero teacher reconstruction norm')
                scores['candidate_relative_output_mse']=scores['candidate_error_sum_squares']/scores['teacher_sum_squares'];scores['incumbent_relative_output_mse']=scores['incumbent_error_sum_squares']/scores['teacher_sum_squares']
            report={'path':path,'expert':expert,'output_rows':out,'input_channels':width,'coverage':coverage,'resources':resource_plan.__dict__,'metrics_proxy':metrics,'factor_counts':factor_counts,'factor_counts_scope':'legacy seed pass; final v2 row choices are in chunk fitting_detail','chunks':chunks};target_reports.append(report);progress['completed_targets'].append({'path':path,'expert':expert,'output_rows':out,'metrics_proxy':metrics});notify('target_fitted')
            del x,dev,factor
        notify('verifying_source_and_transaction')
        after=source_inventory(source,notify)
        if before!=after:raise ValueError('complete indexed BF16 source changed')
        if input_hashes!={p:sha_file(p) for p in input_hashes}:raise ValueError('captured input artifacts changed')
        if sha_file(plan_path)!=progress['plan_sha256']:raise ValueError('recipe plan changed')
        provenance={'format':'gemma_selective_covariance_checkpoint_v1','model':args.model,'recipe':qplan,'plan_sha256':progress['plan_sha256'],'source':str(source),'source_inventory_sha256':before,'template':str(template),'capture_provenance':provenances,'input_sha256':input_hashes,'targets':target_reports,'generated_quality_measured':False,'runtime_speed_measured':False,'all_model_layers_covariance_fitted':False,'metadata_edits':['model-card.json','new covariance-quantization.json'],'inherited_scale_search_report_scope':'Original template recipe/provenance applies to untouched tensors. Selected target/expert replacements are described here.','reconstruction_scope':'FP32 matrix multiplication of BF16-dequantized weights on captured BF16 source inputs; source-layer teacher context, not assembled candidate execution.'}
        card=json.loads((template/'model-card.json').read_text());card.update(name=destination.name,description='Research native Q4 G64 checkpoint with selective covariance plus scale search. Untouched tensors retain the incumbent recipe. Generated coding/cyber quality, runtime and serving memory are unqualified.',quantization={'algorithm':'selective_native_affine_covariance_scale_search','bits':4,'group_size':64,'mode':'affine','scale_search':True,'refinement_steps':qplan['refinement_steps']});transaction.publish(provenance,card)
    notify('complete_checkpoint_unbenchmarked',output=str(destination),seconds_conversion=time.monotonic()-started,provenance_sha256=sha_file(destination/'covariance-quantization.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--model',choices=['a4b','31b'],required=True);p.add_argument('--progress',required=True);a=p.parse_args()
    try:convert(a)
    except BaseException as error:
        progress=Path(a.progress)
        if progress.exists():
            data=json.loads(progress.read_text());data.update(status='failed_conversion_staging_rolled_back',error=repr(error));temp=progress.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(progress)
        raise

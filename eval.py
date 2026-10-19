"""Reproducible experiments with independent GT, ablations, failures and review sheet."""
import bootstrap  # noqa: F401 -- must precede torch imports
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time
import traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import config
import dataset
from core import editor, masker, metrics
from core.instruction_parser import parse

def write_json(path, data):
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [clean(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return 'Infinity' if value>0 else None
        return value
    path.write_text(json.dumps(clean(data),ensure_ascii=False,indent=2),encoding='utf-8')

def write_csv(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['method'],row['action'])].append(row)
    summary=[]
    for (method,action),group in groups.items():
        good=[r for r in group if r['status']=='ok']
        r={'method':method,'action':action,'runs':len(group),'completed':len(good),
           'failed':len(group)-len(good),'unique_images':len({x['image'] for x in group}),
           'gt_evaluated':sum(x.get('evaluation_region')=='human_gt' for x in good)}
        for key in ['global_mae','background_mae','background_psnr','background_changed_fraction',
                    'boundary_mae','mask_iou','clip_score_proxy','background_spatial_lpips',
                    'inference_seconds','peak_vram_mb','lightness_mae','lightness_correlation',
                    'target_color_fraction_diagnostic']:
            values=[float(x[key]) for x in good if isinstance(x.get(key),(int,float)) and math.isfinite(x[key])]
            r[key+'_finite_mean']=float(np.mean(values)) if values else None
            r[key+'_n']=len(values)
        r['background_exact_count']=sum(x.get('background_exact') is True for x in good)
        summary.append(r)
    return summary

def paired_rows(rows,methods):
    groups=defaultdict(list)
    for row in rows:groups[(row['id'],row['seed'])].append(row)
    return [row for group in groups.values()
            if all(any(r['method']==m and r['status']=='ok' for r in group) for m in methods)
            for row in group if row['method'] in methods and row['status']=='ok']

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--settings',default=str(config.ROOT/'experiment.json'))
    p.add_argument('--limit',type=int);p.add_argument('--methods',nargs='+',choices=list(editor.METHODS))
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--learned',action='store_true')
    a=p.parse_args();settings=json.loads(Path(a.settings).read_text(encoding='utf-8'))
    if a.methods:settings['methods']=a.methods
    if a.seeds:settings['seeds']=a.seeds
    if a.learned:settings['learned_metrics']=True
    report=dataset.validate(split=a.split,require_gt=a.split=='test')
    if report['errors']:raise ValueError('\n'.join(report['errors']))
    for msg in report['warnings']:print('WARNING:',msg,flush=True)
    if a.split=='test':
        if a.limit:raise ValueError('正式测试须跑完整冻结集合，不允许 --limit。')
        dataset.check_lock(settings)
    items=[it for it in dataset.load_items() if it['split']==a.split]
    if a.limit:items=items[:a.limit]
    torch,device=editor.runtime()
    run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000)
    out=config.OUTPUT_DIR/'experiments'/run_id;out.mkdir(parents=True)
    versions={}
    for n in ['torch','torchvision','diffusers','transformers','gradio','numpy','Pillow','lpips']:
        try:versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:versions[n]=None
    source_hashes=dataset.source_hashes()
    write_json(out/'run.json',{'split':a.split,'settings':settings,'dataset':report,
        'manifest_sha256':dataset.digest(dataset.MANIFEST),'source_hashes':source_hashes,
        'versions':versions,'python':platform.python_version(),'os':platform.platform(),
        'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
        'device':device,'model_directory':str(config.MODEL_DIR),
        'timing_note':'inference_seconds excludes shared parse/mask preparation and per-engine warmup when enabled; includes transfer, generation and blending',
        'lpips_definition':'GT-weighted spatial LPIPS; receptive fields may cross ROI boundaries',
        'clip_note':'full image/text similarity proxy; not an edit success rate'})
    rows,review,warmed=[],[],set()
    for item in items:
        sample=out/item['id'];sample.mkdir()
        try:
            begin=time.perf_counter()
            with Image.open(dataset.resolve(item['image'])) as im:image=editor.prepare_image(im)
            image.save(sample/'input.png')
            intent=parse(item['instruction'])
            if intent.action != item['action']:
                raise ValueError(f"解析动作 {intent.action} 与登记动作 {item['action']} 不一致")
            bundle=None;bundle_error=None
            try:
                bundle=editor.get_bundle(image,intent,threshold=settings['mask_threshold'],
                    dilate=settings['dilate'],feather=settings['feather'],backend=settings.get('mask_backend','clipseg'))
            except Exception as exc:
                bundle_error=f'{type(exc).__name__}: {exc}'
                (sample/'mask_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            preprocessing_seconds=time.perf_counter()-begin
            if bundle is not None:
                Image.fromarray(bundle['raw']).save(sample/'mask_raw.png')
                Image.fromarray(bundle['hard']).save(sample/'mask_hard.png')
                Image.fromarray(np.rint(bundle['soft']*255).astype(np.uint8)).save(sample/'mask_soft.png')
                masker.overlay(image,bundle['hard']).save(sample/'mask_overlay.png')
                for instance_index,component in enumerate(bundle['components']):
                    Image.fromarray(component.astype(np.uint8)*255).save(sample/f'instance_{instance_index:02d}.png')
                if bundle['prob'] is not None:np.save(sample/'probability.npy',bundle['prob'])
            gt=None
            if item.get('gt_mask') and item.get('gt_reviewed') is True:
                with Image.open(dataset.resolve(item['gt_mask'])) as im:
                    gt=np.asarray(im.convert('L').resize(image.size,Image.Resampling.NEAREST))
                Image.fromarray(gt).save(sample/'gt_reference.png')
            manual=None
            if item.get('manual_mask'):
                with Image.open(dataset.resolve(item['manual_mask'])) as im:manual=im.convert('L').copy()
            write_json(sample/'sample.json',{'item':item,'intent':intent.to_dict(),
                'mask':{k:bundle[k] for k in ['source','area_fraction','component_count']} if bundle else None,
                'mask_error':bundle_error,
                'instances':bundle.get('instances') if bundle else None,
                'preprocessing_seconds':preprocessing_seconds})
            for seed in settings['seeds']:
                cache={}
                for method in settings['methods']:
                    row={'id':item['id'],'image':item['image'],'action':item['action'],
                         'difficulty':item.get('difficulty',''),'method':method,'seed':seed,
                         'preprocessing_seconds':preprocessing_seconds}
                    try:
                        if bundle is None and method not in {'ip2p','oracle'}:
                            raise ValueError(f'自动分割失败: {bundle_error}')
                        if method=='oracle' and manual is None:
                            raise ValueError('oracle 需要清单中的 manual_mask。')
                        kwargs={k:settings[k] for k in ['steps','guidance','strength','dilate','feather']}
                        kwargs.update(mask_threshold=settings['mask_threshold'],intent=intent,prepared=True,seed=seed)
                        engine='lab_chroma' if method=='quality' and intent.action=='color' else ('ip2p' if method in {'ip2p','ip2p_mask'} else 'inpaint')
                        if settings.get('warmup') and engine!='lab_chroma' and engine not in warmed:
                            warm_method='ip2p' if engine=='ip2p' else ('oracle' if method=='oracle' else 'ours')
                            editor.edit(image,item['instruction'],warm_method,
                                        bundle=None if warm_method=='oracle' else bundle,
                                        manual_mask=manual if warm_method=='oracle' else None,**kwargs)
                            warmed.add(engine)
                        # Reuse identical raw generations, isolating final compositing effects.
                        equivalent={'inpaint_raw':'ours','inpaint_hard':'ours','ip2p_mask':'ip2p'}.get(method)
                        if method=='quality' and intent.action!='color':equivalent='ours'
                        if equivalent in cache:
                            ref=cache[equivalent];result=dict(ref)
                            soft=(bundle['hard']>127).astype(np.float32) if method=='inpaint_hard' else bundle['soft']
                            start=time.perf_counter()
                            if method!='quality':
                                result['result']=ref['raw'] if method=='inpaint_raw' else editor.blend(image,ref['raw'],soft)
                            result['time']=None
                            row['generation_reused_from']=equivalent
                            row['compositing_seconds']=time.perf_counter()-start
                        else:
                            result=editor.edit(image,item['instruction'],method,
                                bundle=None if method=='oracle' else bundle,
                                manual_mask=manual if method=='oracle' else None,**kwargs)
                        cache[method]=result
                        prefix=f'{method}_seed{seed}'
                        result['result'].save(sample/(prefix+'.png'))
                        result['raw'].save(sample/(prefix+'_raw.png'))
                        if result['mask'] is not None:
                            Image.fromarray(result['mask']).save(sample/(prefix+'_mask.png'))
                        metric_mask=bundle['hard'] if method=='quality' and intent.action=='color' else result['mask']
                        values=metrics.evaluate(image,result['result'],item['eval_text'],
                                                None if method=='ip2p' else metric_mask,gt,
                                                settings.get('learned_metrics',False))
                        if intent.action=='color' and bundle is not None:
                            from core.appearance import diagnostics
                            values.update(diagnostics(image,result['result'],gt if gt is not None else bundle['raw'],intent.new_content))
                            values['color_diagnostics_roi']='human_gt' if gt is not None else 'predicted_region'
                        row.update(status='ok',inference_seconds=result['time'],peak_vram_mb=result['peak_vram_mb'],
                                   effective_strength=result['strength'],engine=result.get('engine'),
                                   effective_steps=result.get('effective_steps'),**values)
                        review.append({'id':item['id'],'method':method,'seed':seed,'image':item['image'],
                            'instruction':item['instruction'],'result':str((sample/(prefix+'.png')).relative_to(out)),
                            'semantic_success':'','count_preserved':'','identity_preserved':'','boundary_natural':'',
                            'failure_type':'','reviewer':'','notes':''})
                    except Exception as exc:
                        row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                        (sample/f'{method}_seed{seed}_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                    rows.append(row)
                    write_csv(out/'metrics_detail.csv',rows)
                    write_json(sample/f'{method}_seed{seed}.json',row)
                    print(item['id'],method,seed,row['status'],row.get('error',''),flush=True)
        except Exception as exc:
            (sample/'preprocessing_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            for seed in settings['seeds']:
                for method in settings['methods']:
                    rows.append({'id':item['id'],'image':item['image'],'action':item['action'],
                                 'method':method,'seed':seed,'status':'failed',
                                 'error':f'preprocessing: {type(exc).__name__}: {exc}'})
            print(item['id'],'PREPROCESSING FAILED',exc,flush=True)
    write_csv(out/'metrics_detail.csv',rows)
    write_csv(out/'metrics_summary.csv',summarize(rows))
    paired=paired_rows(rows,settings['methods'])
    write_csv(out/'metrics_paired_summary.csv',summarize(paired))
    write_json(out/'coverage.json',{'requested_image_seed_pairs':len(items)*len(settings['seeds']),
        'all_methods_completed_pairs':len({(r['id'],r['seed']) for r in paired}),
        'note':'paired summary uses the same completed cases for every method; failures remain in the full summary'})
    write_csv(out/'human_review.csv',review,fields=['id','method','seed','image','instruction','result',
        'semantic_success','count_preserved','identity_preserved','boundary_natural','failure_type','reviewer','notes'])
    write_json(out/'completion.json',{'completed':sum(r['status']=='ok' for r in rows),
                                     'failed':sum(r['status']!='ok' for r in rows)})
    print('Results:',out,flush=True)
    raise SystemExit(1 if any(r['status']!='ok' for r in rows) else 0)

if __name__=='__main__':main()

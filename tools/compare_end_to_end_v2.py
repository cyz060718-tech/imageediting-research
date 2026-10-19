"""Matched original-parameter CLIPSeg baseline vs the new quality/SAM pipeline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: F401
import time,json
import argparse
import numpy as np
from PIL import Image
import config,dataset
from core import editor,masker,metrics,appearance
from core.instruction_parser import parse
from eval import write_csv,write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-baseline',action='store_true')
    args=parser.parse_args()
    out=config.OUTPUT_DIR/'quality_release_v2'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
    previous=config.OUTPUT_DIR/'experiments/20261018_200624_586500'
    previous_run=json.loads((previous/'run.json').read_text(encoding='utf-8')) if (previous/'run.json').exists() else None
    expected={'steps':30,'guidance':9.0,'dilate':15,'feather':21,'mask_threshold':.4}
    can_reuse=args.reuse_baseline and previous_run is not None and all(previous_run['settings'].get(k)==v for k,v in expected.items())
    rows=[]
    write_json(out/'run.json',{'split':'dev','settings':expected,'seed':42,
        'baseline':'CLIPSeg + original SD prompt/parameters + blend',
        'candidate':'GroundingDINO+SAM, native-geometry recolor, original single-pass SD for replace/remove',
        'source_hashes':dataset.source_hashes(),'manifest_sha256':dataset.digest(dataset.MANIFEST),
        'note':'Baseline reused only when prepared pixels and settings exactly match; no test images used.'})
    for item in dataset.load_items():
        if item['split']!='dev':continue
        with Image.open(dataset.resolve(item['image'])) as im:image=editor.prepare_image(im)
        intent=parse(item['instruction']);sub=out/item['id'];sub.mkdir();image.save(sub/'input.png')
        gt=None
        if item.get('gt_mask'):
            with Image.open(dataset.resolve(item['gt_mask'])) as im:gt=np.asarray(im.convert('L').resize(image.size,Image.Resampling.NEAREST))
        # Build a common diagnostic ROI before comparing the two image methods.
        new_bundle=None;new_error=None
        try:new_bundle=editor.get_bundle(image,intent,backend='grounded_sam')
        except Exception as exc:new_error=str(exc)
        for variant in ['original','quality_v2']:
            row={'id':item['id'],'action':item['action'],'variant':variant,'gt_available':gt is not None,'seed':42}
            try:
                reused=False
                if variant=='original':
                    old=previous/item['id']
                    if can_reuse and (old/'ours_seed42.png').exists() and (old/'sample.json').exists():
                        with Image.open(old/'input.png') as im:old_input=np.asarray(im.convert('RGB'))
                        saved_intent=json.loads((old/'sample.json').read_text(encoding='utf-8'))['intent']
                        reused=old_input.shape==np.asarray(image).shape and np.array_equal(old_input,np.asarray(image)) and saved_intent==intent.to_dict()
                    if reused:
                        with Image.open(old/'ours_seed42.png') as im:edited=im.convert('RGB')
                        with Image.open(old/'mask_raw.png') as im:raw_mask=np.asarray(im.convert('L'))
                        with Image.open(old/'mask_hard.png') as im:hard=np.asarray(im.convert('L'))
                        row['reused_baseline']=str(old/'ours_seed42.png')
                        engine='sd_inpaint';details=None
                    else:
                        bundle=editor.get_bundle(image,intent,backend='clipseg')
                        result=editor.edit(image,item['instruction'],'ours',bundle=bundle,intent=intent,prepared=True,
                                           mask_backend='clipseg',steps=30,guidance=9.,seed=42)
                        edited=result['result'];raw_mask=bundle['raw'];hard=bundle['hard'];engine=result['engine'];details=None
                else:
                    if new_bundle is None:raise ValueError(new_error)
                    result=editor.edit(image,item['instruction'],'quality',bundle=new_bundle,intent=intent,prepared=True,
                                       mask_backend='grounded_sam',steps=30,guidance=9.,seed=42)
                    edited=result['result'];raw_mask=new_bundle['raw'];hard=new_bundle['hard'];engine=result['engine']
                    details={'instances':new_bundle['instances'],'engine_info':result.get('engine_info')}
                    for j,component in enumerate(new_bundle['components']):
                        Image.fromarray(component.astype(np.uint8)*255).save(sub/f'instance_{j:02d}.png')
                edited.save(sub/(variant+'.png'));Image.fromarray(raw_mask).save(sub/(variant+'_mask.png'))
                row.update(status='ok',engine=engine,localization_iou=masker.mask_iou(raw_mask,gt) if gt is not None else None,
                           **metrics.evaluate(image,edited,item['eval_text'],hard,gt))
                if intent.action=='color':
                    roi=gt if gt is not None else (new_bundle['raw'] if new_bundle else raw_mask)
                    row.update(appearance.diagnostics(image,edited,roi,intent.new_content))
                    row['color_roi']='human_gt' if gt is not None else 'predicted_region'
                write_json(sub/(variant+'_details.json'),details or {})
            except Exception as exc:
                row.update(status='failed',error=str(exc),localization_iou=0. if gt is not None else None)
            rows.append(row);write_json(sub/(variant+'.json'),row);write_csv(out/'comparison.csv',rows)
            print(item['id'],variant,row['status'],row.get('error',''),flush=True)
    summary=[]
    common_colors={r['id'] for r in rows if r['variant']=='original' and r['action']=='color' and r['status']=='ok' and r['gt_available']}
    common_colors &= {r['id'] for r in rows if r['variant']=='quality_v2' and r['action']=='color' and r['status']=='ok' and r['gt_available']}
    for variant in ['original','quality_v2']:
        group=[r for r in rows if r['variant']==variant]
        gt_rows=[r for r in group if r['gt_available']]
        colors=[r for r in gt_rows if r['id'] in common_colors]
        summary.append({'variant':variant,'requested':len(group),'completed':sum(r['status']=='ok' for r in group),
            'gt_cases':len(gt_rows),'mean_localization_iou_with_misses':float(np.mean([r['localization_iou'] for r in gt_rows])),
            'paired_color_gt_n':len(colors),
            'color_lightness_mae':float(np.mean([r['lightness_mae'] for r in colors])) if colors else None,
            'color_lightness_correlation':float(np.mean([r['lightness_correlation'] for r in colors])) if colors else None})
    write_csv(out/'summary.csv',summary);write_json(out/'completion.json',summary)
    print('OUTPUT',out,flush=True)

if __name__=='__main__':main()

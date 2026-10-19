"""Compare CLIPSeg and GroundingDINO+SAM on dev, counting misses as zero IoU."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: F401
import time
import numpy as np
from PIL import Image
import config,dataset
from core import editor,masker,grounded_mask,appearance
from core.instruction_parser import parse
from eval import write_csv,write_json

def main():
    out=config.OUTPUT_DIR/'mask_upgrade_v2'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
    rows=[]
    for it in dataset.load_items():
        if it['split']!='dev':continue
        with Image.open(dataset.resolve(it['image'])) as im:image=editor.prepare_image(im)
        intent=parse(it['instruction']);sub=out/it['id'];sub.mkdir();image.save(sub/'input.png')
        gt=None
        if it.get('gt_mask'):
            with Image.open(dataset.resolve(it['gt_mask'])) as im:gt=np.asarray(im.convert('L').resize(image.size,Image.Resampling.NEAREST))
        for backend in ['clipseg','grounded_sam']:
            t0=time.perf_counter();row={'id':it['id'],'backend':backend,'gt_available':gt is not None}
            try:
                b=editor.get_bundle(image,intent,backend=backend)
                Image.fromarray(b['raw']).save(sub/(backend+'_mask.png'))
                masker.overlay(image,b['raw']).save(sub/(backend+'_overlay.png'))
                row.update(status='ok',instances=b['component_count'],
                    raw_iou=masker.mask_iou(b['raw'],gt) if gt is not None else None,
                    hard_iou=masker.mask_iou(b['hard'],gt) if gt is not None else None)
                if it['action']=='color':
                    result,alpha,_=appearance.recolor(image,b['raw'],intent.new_content)
                    result.save(sub/(backend+'_recolor.png'))
                write_json(sub/(backend+'.json'),{'instances':b.get('instances'),'mask_source':b['source']})
            except Exception as exc:
                row.update(status='failed',raw_iou=0. if gt is not None else None,hard_iou=0. if gt is not None else None,error=str(exc))
            row['seconds']=time.perf_counter()-t0;rows.append(row)
            write_csv(out/'mask_detail.csv',rows)
            print(it['id'],backend,row['status'],round(row['seconds'],2),row.get('raw_iou'),row.get('error',''),flush=True)
    for backend in ['clipseg','grounded_sam']:
        group=[r for r in rows if r['backend']==backend and r['gt_available']]
        print(backend,'GT n',len(group),'mean raw IoU',float(np.mean([r['raw_iou'] for r in group])),
              'completed',sum(r['status']=='ok' for r in rows if r['backend']==backend),flush=True)
    write_json(out/'run.json',{'split':'dev','grounded_threshold':.25,'text_threshold':.25,
        'source_hashes':dataset.source_hashes(),'note':'No GT used during prediction; failures retained as zero IoU.'})
    print('OUTPUT',out,flush=True)

if __name__=='__main__':main()

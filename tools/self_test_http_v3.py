"""Exercise the shipped UI handlers via HTTP on all dev cases; never reads test images."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse,json,shutil,time
import numpy as np
from PIL import Image
from gradio_client import Client,handle_file
import config,dataset


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:7863/')
    parser.add_argument('--actions',nargs='+',choices=['color','remove','replace'],default=['color','remove','replace'])
    parser.add_argument('--tag',default='final_http')
    parser.add_argument('--ids',nargs='+')
    args=parser.parse_args()
    root=config.OUTPUT_DIR/'self_test_v3'/args.tag
    root.mkdir(parents=True,exist_ok=True)
    (root/'run.json').write_text(json.dumps({'split':'dev','url':args.url,
        'source_hashes':dataset.source_hashes(),'note':'generated does not mean semantic success'},
        ensure_ascii=False,indent=2),encoding='utf-8')
    client=Client(args.url,verbose=False)
    rows=[]
    for item in dataset.load_items():
        if item['split']!='dev' or item['action'] not in args.actions:continue
        if args.ids and item['id'] not in args.ids:continue
        row={'id':item['id'],'action':item['action'],'instruction':item['instruction']}
        start=time.perf_counter()
        try:
            image=str(dataset.resolve(item['image']))
            preview=client.predict(handle_file(image),item['instruction'],None,.4,15,21,False,
                'quality','grounded_sam','auto',api_name='/preview')
            response=client.predict(handle_file(image),item['instruction'],None,'quality',30,9,0,42,
                .4,15,21,False,'grounded_sam','auto',api_name='/run')
            record=response[3];source=Path(record['output_directory']);dest=root/item['id'];dest.mkdir(exist_ok=True)
            for name in ('input.png','result.png','mask_raw.png','mask_hard.png','mask_overlay.png','soft_mask.npy','record.json'):
                shutil.copy2(source/name,dest/name)
            if (source/'candidates').exists():
                shutil.copytree(source/'candidates',dest/'candidates',dirs_exist_ok=True)
            original=np.asarray(Image.open(dest/'input.png').convert('RGB'))
            edited=np.asarray(Image.open(dest/'result.png').convert('RGB'))
            alpha=np.load(dest/'soft_mask.npy')
            assert original.shape==edited.shape
            protected=alpha==0
            assert np.array_equal(original[protected],edited[protected]),'Protected pixels changed'
            row.update(status='generated',seconds=time.perf_counter()-start,engine=record['engine'],
                selected_regions=preview[1]['selected_regions'],mask_source=record['mask_source'],
                size=[original.shape[1],original.shape[0]],protected_pixels_exact=True,
                automatic_check=(record.get('engine_info') or {}).get('automatic_check'),
                peak_vram_mb=record['peak_vram_mb'],result_directory=str(dest))
        except Exception as exc:
            row.update(status='runtime_failed',error=str(exc),seconds=time.perf_counter()-start)
        rows.append(row)
        (root/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print('HTTP_SELF_TEST',item['id'],row['status'],round(row['seconds'],2),
            (row.get('automatic_check') or {}).get('status',''),flush=True)
    print('HTTP_SELF_TEST_COMPLETE',len(rows),sum(row['status']=='generated' for row in rows),root,flush=True)
    if any(row['status']!='generated' for row in rows):sys.exit(1)

if __name__=='__main__':main()

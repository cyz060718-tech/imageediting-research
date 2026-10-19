"""Create reproducible independent COCO dev/test samples with provider human GT.

Fetch only the compressed val annotation member through HTTP Range, then selected
images, rather than the full image archive. Paths and licenses are recorded.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import concurrent.futures
import hashlib
import io
import json
import random
import struct
import time
import urllib.request
import zlib
from collections import defaultdict,Counter
import numpy as np
import cv2
from PIL import Image
import config,dataset

SOURCE='https://s3.amazonaws.com/images.cocodataset.org/'
ZIP_URL=SOURCE+'annotations/annotations_trainval2017.zip'
SEED=20261018
COLOR={'car','bus','truck','bicycle','motorcycle','umbrella','handbag','tie','suitcase','bottle','cup','bowl','chair','couch','vase','backpack','boat','train'}
REMOVE={'stop sign','traffic light','bench','chair','bottle','cup','umbrella','handbag','backpack','cat','dog','suitcase','vase','potted plant'}
REPLACE={'cat':'dog','dog':'cat','horse':'zebra','sheep':'cow','cow':'horse','bottle':'vase','cup':'bowl','chair':'armchair','couch':'bench'}

def plural(noun):
    if noun=='sheep':return noun
    return noun+('es' if noun.endswith(('s','ch','sh')) else 's')

def get(url,start=None,end=None):
    headers={'User-Agent':'ImageEditingResearch/1.0'}
    if start is not None:headers['Range']=f'bytes={start}-'+(str(end) if end is not None else '')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as r:
                if start is not None and r.status!=206:raise ValueError('Server did not honor byte range')
                return r.read()
        except Exception:
            if attempt==2:raise
            time.sleep(1)

def annotations():
    cache=dataset.DATA/'source_cache';cache.mkdir(exist_ok=True)
    path=cache/'instances_val2017.json'
    if not path.exists():
        # ZIP EOCD and central directory use little-endian ZIP structures.
        with urllib.request.urlopen(urllib.request.Request(ZIP_URL,headers={'Range':'bytes=-65536'}),timeout=30) as r:
            if r.status!=206:raise ValueError('Byte ranges unavailable')
            tail=r.read();total=int(r.headers['Content-Range'].split('/')[-1])
        eocd=tail.rfind(b'PK\x05\x06')
        if eocd<0:raise ValueError('Missing ZIP directory')
        _,_,_,_,_,central_size,central_start,_=struct.unpack_from('<4s4H2IH',tail,eocd)
        central=tail[central_start-(total-len(tail)):central_start-(total-len(tail))+central_size] if central_start>=total-len(tail) else get(ZIP_URL,central_start,central_start+central_size-1)
        pos=0;found=None
        while pos<len(central):
            if central[pos:pos+4]!=b'PK\x01\x02':raise ValueError('Bad ZIP central directory')
            compression=struct.unpack_from('<H',central,pos+10)[0]
            crc,compressed,uncompressed=struct.unpack_from('<III',central,pos+16)
            name_len,extra_len,comment_len=struct.unpack_from('<HHH',central,pos+28)
            offset=struct.unpack_from('<I',central,pos+42)[0]
            name=central[pos+46:pos+46+name_len].decode()
            if name=='annotations/instances_val2017.json':found=(compression,crc,compressed,uncompressed,offset);break
            pos+=46+name_len+extra_len+comment_len
        if found is None:raise ValueError('Validation annotations not found')
        compression,crc,compressed,uncompressed,offset=found
        header=get(ZIP_URL,offset,offset+29);name_len,extra_len=struct.unpack_from('<HH',header,26)
        start=offset+30+name_len+extra_len;payload=get(ZIP_URL,start,start+compressed-1)
        raw=zlib.decompress(payload,-15) if compression==8 else payload
        if len(raw)!=uncompressed or zlib.crc32(raw)!=crc:raise ValueError('Annotation ZIP integrity check failed')
        path.write_bytes(raw)
    return json.loads(path.read_text(encoding='utf-8'))

def dhash(image):
    a=np.asarray(image.convert('L').resize((9,8)))
    return np.packbits(a[:,1:]>a[:,:-1]).tobytes()

def main():
    import pycocotools.mask as coco_mask
    existing=dataset.load_items()
    if any(it.get('dataset')=='COCO val2017' for it in existing):
        raise ValueError('COCO subset already registered; refusing duplicate import.')
    data=annotations();images={im['id']:im for im in data['images']}
    categories={c['id']:c['name'] for c in data['categories']};licenses={x['id']:x for x in data['licenses']}
    by_image=defaultdict(list)
    for ann in data['annotations']:by_image[ann['image_id']].append(ann)
    candidates=defaultdict(list)
    for image_id,anns in by_image.items():
        if image_id==39769:continue  # existing two-cat example
        im=images[image_id];by_category=defaultdict(list)
        for ann in anns:by_category[categories[ann['category_id']]].append(ann)
        for category,objects in by_category.items():
            if any(a.get('iscrowd') or not isinstance(a['segmentation'],list) for a in objects):continue
            if not 1<=len(objects)<=4:continue
            area=sum(a['area'] for a in objects)/(im['width']*im['height'])
            if not .015<=area<=.50:continue
            if len(objects)>1:difficulty='multiple_instances'
            elif area<.05:difficulty='small_target'
            elif len(anns)>=8:difficulty='complex_background'
            else:difficulty='simple'
            for action,allowed in [('color',COLOR),('remove',REMOVE),('replace',REPLACE)]:
                if category in allowed:candidates[(action,difficulty)].append((image_id,category,objects))
    rng=random.Random(SEED)
    for values in candidates.values():rng.shuffle(values)
    used=set();chosen=[]
    # Add 17 dev to the 3 existing examples = 20 dev. Preserve 120 unseen test.
    for split,counts in [('dev',{'color':6,'remove':6,'replace':5}),('test',{'color':40,'remove':40,'replace':40})]:
        for action,n in counts.items():
            for index in range(n):
                difficulty=['simple','complex_background','small_target','multiple_instances'][index%4]
                options=candidates[(action,difficulty)]
                candidate=next((x for x in options if x[0] not in used),None)
                if candidate is None:raise ValueError(f'Insufficient candidates for {action} {difficulty}')
                used.add(candidate[0]);chosen.append((split,action,difficulty,*candidate))
    images_dir=dataset.DATA/'images'/'coco';masks_dir=dataset.DATA/'masks'/'coco'
    images_dir.mkdir(parents=True,exist_ok=True);masks_dir.mkdir(parents=True,exist_ok=True)
    def download(task):
        split,action,difficulty,image_id,category,objects=task
        im=images[image_id];file=images_dir/im['file_name']
        if not file.exists():file.write_bytes(get(SOURCE+'val2017/'+im['file_name']))
        with Image.open(file) as picture:
            picture=picture.convert('RGB');picture.load()
            if picture.size!=(im['width'],im['height']):raise ValueError('Image size disagrees with metadata')
            perceptual=dhash(picture)
        rles=[]
        for ann in objects:rles.extend(coco_mask.frPyObjects(ann['segmentation'],im['height'],im['width']))
        mask=coco_mask.decode(coco_mask.merge(rles));mp=masks_dir/(file.stem+'.png')
        Image.fromarray(mask.astype(np.uint8)*255).save(mp)
        noun=plural(category) if len(objects)>1 else category
        original_blue_fraction=None
        if action=='color':
            hsv=cv2.cvtColor(np.asarray(picture),cv2.COLOR_RGB2HSV)
            original_blue_fraction=float(((hsv[...,0]>=90)&(hsv[...,0]<=140)&(hsv[...,1]>64))[mask>0].mean())
            color='red' if original_blue_fraction>.35 else 'blue'
            instruction=f'change the {noun} to {color}';target=f'a photograph of {color} {noun} in the original scene'
        elif action=='remove':instruction=f'remove the {noun}';target=f'a natural scene without {noun}, with the background restored'
        else:
            replacement=plural(REPLACE[category]) if len(objects)>1 else REPLACE[category]
            instruction=f'replace the {noun} with {replacement}'
            target=f'a photograph of {len(objects)} {replacement} in the original scene'
        return {'id':f'coco_{split}_{action}_{image_id:012d}','image':str(file.relative_to(dataset.DATA)).replace('\\','/'),
                'instruction':instruction,'action':action,'split':split,'eval_text':target,
                'source_group':f'coco_val2017_{image_id}','difficulty':difficulty,
                'gt_mask':str(mp.relative_to(dataset.DATA)).replace('\\','/'),
                'gt_reviewed':True,'gt_review_basis':'COCO provider human instance annotations, CRC-verified official source; not newly hand-labeled locally',
                'local_human_review':False,'dataset':'COCO val2017','coco_image_id':image_id,
                'coco_annotation_ids':[a['id'] for a in objects],'target_category':category,
                'expected_target_count':len(objects),'source_url':SOURCE+'val2017/'+im['file_name'],
                'original_blue_fraction_diagnostic':original_blue_fraction,
                'flickr_url':im.get('flickr_url',''),'license':licenses[im['license']],
                'notes':'Target means all annotated instances of this category. Mask is the union of independent provider polygons. Deletion does not include unannotated supports or shadows.',
                '_dhash':perceptual.hex()}
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(download,task) for task in chosen]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result());print(f'downloaded {len(results)}/{len(chosen)}',flush=True)
    results.sort(key=lambda x:x['id'])
    known=[]
    for it in existing:
        with Image.open(dataset.resolve(it['image'])) as im:known.append((it['id'],dhash(im)))
    for it in results:
        h=bytes.fromhex(it.pop('_dhash'))
        for ident,other in known:
            distance=sum(bin(x^y).count('1') for x,y in zip(h,other))
            if distance<=3:raise ValueError(f'Potential near duplicate: {it["id"]} / {ident}; inspect before registering')
        known.append((it['id'],h))
    combined=existing+results;report=dataset.validate(combined)
    if report['errors']:raise ValueError('\n'.join(report['errors']))
    backup=dataset.DATA/'manifest.before_coco.json'
    if backup.exists():raise FileExistsError(backup)
    backup.write_bytes(dataset.MANIFEST.read_bytes())
    dataset.MANIFEST.write_text(json.dumps(combined,ensure_ascii=False,indent=2),encoding='utf-8')
    record={'seed':SEED,'annotation_source':ZIP_URL,'annotations_sha256':dataset.digest(dataset.DATA/'source_cache/instances_val2017.json'),
            'gt_definition':'COCO human polygon union for all non-crowd target instances; no prediction-derived labels',
            'selection':'unique image IDs, per-task strata, pixel and source-group validation, dHash duplicate screen',
            'counts':dict(Counter(x['split'] for x in combined)),
            'local_human_review':False,'provider_annotations_used_as_reference':True}
    (dataset.DATA/'coco_provenance.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

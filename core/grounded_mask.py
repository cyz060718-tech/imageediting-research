"""Local Grounding DINO + SAM instance masks, independently evaluated against GT."""
import time
import hashlib
import threading
from collections import OrderedDict
import numpy as np
import cv2
from pathlib import Path
import config
from core import masker

_cache={}
_predictions=OrderedDict()
_prediction_lock=threading.RLock()

def remove_small_islands(component):
    """Remove small disconnected leaks within one SAM object, not other instances."""
    count,labels,stats,_=cv2.connectedComponentsWithStats(component.astype(np.uint8),8)
    if count<=1:
        return component
    largest=int(stats[1:,cv2.CC_STAT_AREA].max())
    keep=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=largest*.05]
    return np.isin(labels,keep).astype(np.uint8)

def deduplicate_instances(components,details):
    """Discard almost-contained duplicate masks of the same queried category."""
    kept=[]
    for i in sorted(range(len(components)),key=lambda n:int(components[n].sum()),reverse=True):
        area=int(components[i].sum())
        if any(np.logical_and(components[i],components[j]).sum()/max(area,1)>.95 for j in kept):
            continue
        kept.append(i)
    kept.sort()
    return [components[i] for i in kept],[details[i] for i in kept]

def available():
    return all((config.EXTRA_MODEL_DIR/name/'model.safetensors').is_file()
               for name in config.EXTRA_MODEL_REPOS)

def load():
    if not _cache:
        import torch
        torch.set_num_threads(min(8,torch.get_num_threads()))
        from transformers import AutoModelForZeroShotObjectDetection,AutoProcessor,SamModel,SamProcessor
        detector=config.model_path('grounding_dino');sam=config.model_path('sam')
        _cache['detector_processor']=AutoProcessor.from_pretrained(detector,local_files_only=True)
        _cache['detector']=AutoModelForZeroShotObjectDetection.from_pretrained(detector,local_files_only=True).eval()
        _cache['sam_processor']=SamProcessor.from_pretrained(sam,local_files_only=True)
        _cache['sam']=SamModel.from_pretrained(sam,local_files_only=True).eval()
    return _cache

def detect_boxes(image,text,threshold=.25):
    """CPU-only diagnostic detector; its result is not a semantic success score."""
    import torch
    from torchvision.ops import nms
    with _prediction_lock,torch.inference_mode():
        models=load();processor=models['detector_processor']
        inputs=processor(images=image,text=text.lower().strip().rstrip('.')+'.',return_tensors='pt')
        output=models['detector'](**inputs)
        found=processor.post_process_grounded_object_detection(output,inputs.input_ids,
            threshold=threshold,text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
        if not len(found['boxes']):return []
        indices=nms(found['boxes'],found['scores'],.5)[:16]
        return [{'box':found['boxes'][i].tolist(),'score':float(found['scores'][i])} for i in indices]

def _generate(image,text,dilate=15,feather=21,box_threshold=.25,protect_contents=False):
    import torch
    from torchvision.ops import nms
    if not text.strip():raise ValueError('目标为空。')
    t0=time.perf_counter();models=load();processor=models['detector_processor']
    query=text.lower().strip().rstrip('.')+'.'
    with torch.inference_mode():
        inputs=processor(images=image,text=query,return_tensors='pt')
        outputs=models['detector'](**inputs)
        detection=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
            threshold=box_threshold,text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
        boxes,scores=detection['boxes'],detection['scores']
        boxes[:,0::2].clamp_(0,image.width);boxes[:,1::2].clamp_(0,image.height)
        keep=(boxes[:,2]-boxes[:,0]>2)&(boxes[:,3]-boxes[:,1]>2)
        boxes,scores=boxes[keep],scores[keep]
        if not len(boxes):raise ValueError('检测器未找到可靠目标，请手绘选区或切换 CLIPSeg。')
        indices=nms(boxes,scores,.5)[:8];boxes,scores=boxes[indices],scores[indices]
        primary_count=len(boxes);protected_boxes=[]
        if protect_contents and text.lower().strip() in {'bowl','bowls','plate','plates','dish','dishes'}:
            protect_inputs=processor(images=image,text='food. banana. ice cream. soup.',return_tensors='pt')
            protect_output=models['detector'](**protect_inputs)
            protection=processor.post_process_grounded_object_detection(protect_output,protect_inputs.input_ids,
                threshold=.30,text_threshold=.25,target_sizes=[(image.height,image.width)])[0]
            if len(protection['boxes']):
                kept=nms(protection['boxes'],protection['scores'],.5)[:6]
                protected_boxes=protection['boxes'][kept].tolist()
        all_boxes=boxes.tolist()+protected_boxes
        sam_inputs=models['sam_processor'](image,input_boxes=[all_boxes],return_tensors='pt')
        output=models['sam'](**sam_inputs)
        masks=models['sam_processor'].image_processor.post_process_masks(output.pred_masks,
            sam_inputs['original_sizes'],sam_inputs['reshaped_input_sizes'])[0]
        predicted_quality=output.iou_scores[0]
        protected_masks=[]
        for j in range(primary_count,len(all_boxes)):
            best=int(predicted_quality[j].argmax())
            protected_masks.append(masks[j,best].cpu().numpy().astype(bool))
        components=[];details=[]
        for i in range(primary_count):
            best=int(predicted_quality[i].argmax());component=masks[i,best].cpu().numpy().astype(np.uint8)
            component=remove_small_islands(component)
            protected_fraction=0.
            if protected_boxes:
                protect=np.zeros((image.height,image.width),bool)
                for content_mask in protected_masks:
                    overlap=content_mask & component.astype(bool)
                    fraction=float(overlap.sum()/max(component.sum(),1))
                    # A loose food box may yield the entire vessel. Ignore that
                    # candidate without discarding useful inner-content masks.
                    if .01<=fraction<.80:
                        protect |= overlap
                proposed=component.astype(bool)&~protect
                retained=float(proposed.sum()/max(component.sum(),1))
                if retained>=.20:
                    protected_fraction=1-retained;component=proposed.astype(np.uint8)
            if component.sum()<16:continue
            components.append(component)
            details.append({'box':boxes[i].tolist(),'detection_score':float(scores[i]),
                            'sam_predicted_iou':float(predicted_quality[i,best]),
                            'protected_content_fraction':protected_fraction})
    components,details=deduplicate_instances(components,details)
    if not components:raise ValueError('SAM 未生成非空选区，请手绘修正。')
    binary=np.maximum.reduce(components)
    if binary.mean()>.85:raise ValueError('选区超过图像的 85%，请手工确认编辑范围。')
    hard,soft=masker.refine_binary(binary,dilate,feather)
    return {'hard':hard,'soft':soft,'raw':binary*255,'prob':None,
        'components':components,'component_count':len(components),'area_fraction':float(binary.mean()),
        'source':'grounding_dino_sam','instances':details,'seconds':time.perf_counter()-t0}

def generate(image,text,dilate=15,feather=21,box_threshold=.25,protect_contents=False):
    started=time.perf_counter()
    key=(hashlib.sha256(image.convert('RGB').tobytes()).hexdigest(),image.size,text,dilate,feather,box_threshold,protect_contents)
    with _prediction_lock:
        if key in _predictions:
            bundle=_predictions.pop(key);_predictions[key]=bundle
            return dict(bundle,cache_hit=True,seconds=time.perf_counter()-started)
        bundle=_generate(image,text,dilate,feather,box_threshold,protect_contents)
        _predictions[key]=bundle
        limit=1 if image.width*image.height>4000000 else 3
        while len(_predictions)>limit:_predictions.popitem(last=False)
        return dict(bundle,cache_hit=False)

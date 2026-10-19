"""Inspectable editing: fixed masks/intent, raw outputs, identity-conservative defaults."""
import gc
import time
import numpy as np
from PIL import Image, ImageOps
import config
from core import masker
from core.instruction_parser import parse

METHODS = {
    'quality': '保真模式：改色保纹理 / 删除修补背景 / 替换局部重绘',
    'ours': 'SD 重绘（原版生成参数）',
    'local_crop': '试验：局部放大重绘（小目标）',
    'inpaint_raw': '消融：相同局部重绘，取消最终融合',
    'inpaint_hard': '消融：相同局部重绘，硬掩码融合',
    'ours_instances': '试验：连通域逐个重绘（不等同于实例分割）',
    'ip2p': '基线：InstructPix2Pix 全图编辑',
    'ip2p_mask': '消融：相同 IP2P 输出 + 掩码融合',
    'oracle': '诊断：人工编辑掩码 + 局部重绘',
}
_pipes = {}

def runtime():
    import torch
    return torch, 'cuda' if torch.cuda.is_available() else 'cpu'

def get_pipe(name):
    torch, device = runtime()
    if name not in _pipes:
        # Offloaded pipelines release their active modules after each request.
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
        cls = StableDiffusionInpaintPipeline if name == 'inpaint' else StableDiffusionInstructPix2PixPipeline
        path = config.model_path(name)
        from pathlib import Path
        kw = {'dtype': torch.float16 if device == 'cuda' else torch.float32,
              'local_files_only': True}
        # Prefer fp16 ONLY when this repository actually contains that variant.
        if any(Path(path).rglob('*.fp16.safetensors')):
            kw['variant'] = 'fp16'
        pipe = cls.from_pretrained(path, **kw)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        if device == 'cuda':
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        _pipes[name] = pipe
    return _pipes[name]

def prepare_image(image, size=config.IMAGE_SIZE):
    image = ImageOps.exif_transpose(image).convert('RGB')
    w,h = image.size
    if min(w,h) < 8:
        raise ValueError('图像宽高至少为 8 像素。')
    scale = size/max(w,h)
    w,h = max(8, int(w*scale)//8*8), max(8, int(h*scale)//8*8)
    return image.resize((w,h), Image.Resampling.LANCZOS)

def prepare_for_method(image,intent,method):
    # Chroma edits and crop-based replacement both retain the native canvas.
    if method=='quality':
        return ImageOps.exif_transpose(image).convert('RGB')
    return prepare_image(image)

def blend(original, edited, soft):
    a,b = np.asarray(original, dtype=np.float32), np.asarray(edited, dtype=np.float32)
    if a.shape != b.shape or soft.shape != a.shape[:2]:
        raise ValueError('融合图像和掩码尺寸不一致。')
    return Image.fromarray(np.clip(np.rint(a*(1-soft[...,None])+b*soft[...,None]),0,255).astype(np.uint8))

def get_bundle(image, intent, manual_mask=None, threshold=config.MASK_THRESHOLD,
               dilate=config.MASK_DILATE, feather=config.MASK_FEATHER,backend='clipseg'):
    if manual_mask is not None:
        return masker.manual_masks(manual_mask, image.size, feather=feather)
    if backend=='grounded_sam':
        from core.grounded_mask import generate
        return generate(image,intent.mask_prompt,dilate,feather,protect_contents=intent.action=='color')
    if backend!='clipseg':raise ValueError(f'未知分割方式 {backend}')
    return masker.generate_bundle(image, intent.mask_prompt, threshold=threshold, dilate=dilate, feather=feather)

def _to_instruction(intent):
    if intent.action == 'remove':
        return f'remove the {intent.target}'
    if intent.action == 'color':
        return f'change the color of the {intent.target} to {intent.new_content}'
    return f'replace the {intent.target} with {intent.new_content}'

def edit(image, instruction, method='ours', steps=config.DEFAULT_STEPS, guidance=config.DEFAULT_GUIDANCE,
         strength=None, seed=42, use_llm=False, mask_threshold=.4,
         dilate=config.MASK_DILATE, feather=config.MASK_FEATHER, manual_mask=None,
         bundle=None, intent=None, prepared=False,mask_backend=config.DEFAULT_MASK_BACKEND):
    if method not in METHODS:
        raise ValueError(f'未知方法 {method}')
    t0 = time.perf_counter()
    if not prepared and manual_mask is not None and manual_mask.size != ImageOps.exif_transpose(image).size:
        raise ValueError('人工编辑掩码须与 EXIF 转正后的原图同尺寸。')
    intent = intent or parse(instruction, use_llm)
    image = image if prepared else prepare_for_method(image,intent,method)
    if intent.action not in {'color','remove','replace'} or not intent.target:
        raise ValueError('当前调试版只接受明确的改色、删除或替换指令，请指明目标。')
    import re
    if manual_mask is None and (bundle is None or bundle.get('source')!='manual') and re.search(r'左边|右边|左侧|右侧|最左|最右|第[一二三四]|\b(left|right|first|second)\b', instruction, re.I):
        raise ValueError('方位/序号选择需要上传人工编辑掩码，避免自动修改所有同类目标。')
    if method == 'oracle' and manual_mask is None:
        raise ValueError('oracle 对照需要单独提供人工编辑掩码 manual_mask，不能自动使用 GT。')
    if method=='quality' and intent.action=='color':
        from core.appearance import recolor
        if bundle is None:
            bundle=get_bundle(image,intent,manual_mask,mask_threshold,dilate,feather,backend=mask_backend)
        result,alpha,info=recolor(image,bundle['raw'],intent.new_content)
        effective=(alpha>0).astype(np.uint8)*255
        effective_bundle=dict(bundle,hard=effective,soft=alpha,source=bundle['source']+'_color_support')
        return {'input':image,'result':result,'raw':result,'mask':effective,
                'mask_overlay':masker.overlay(image,effective),'bundle':effective_bundle,'intent':intent,
                'strength':None,'effective_steps':0,'engine':'lab_chroma','engine_info':info,
                'raw_kind':'deterministic_color_transform','time':time.perf_counter()-t0,
                'peak_vram_mb':None,'device':'cpu'}
    if method=='quality' and intent.action=='remove':
        from core.removal import remove
        if bundle is None:
            bundle=get_bundle(image,intent,manual_mask,mask_threshold,dilate,feather,backend=mask_backend)
        result,raw=remove(image,bundle['hard'],bundle['soft'])
        effective=(bundle['soft']>0).astype(np.uint8)*255
        return {'input':image,'result':result,'raw':raw,'mask':effective,
            'mask_overlay':masker.overlay(image,effective),'bundle':dict(bundle,hard=effective),
            'intent':intent,'strength':None,'time':time.perf_counter()-t0,
            'peak_vram_mb':None,'device':'cpu','engine':'lama_background_repair','effective_steps':0,
            'engine_info':{'native_canvas':True,'model':'LaMa','note':'Only the selected area is removed; supports outside it remain.'},
            'raw_kind':'background_repair_output'}
    torch, device = runtime()
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if bundle is None and method != 'ip2p':
        bundle = get_bundle(image, intent, manual_mask, mask_threshold, dilate, feather,backend=mask_backend)
    hard,soft = (bundle['hard'],bundle['soft']) if bundle else (None,None)
    # The dev sweep showed .65/.85 failed to recolor this inpainting model.
    strength = 1. if strength is None else float(strength)
    if not 0 < strength <= 1 or int(steps*strength) < 1:
        raise ValueError('编辑强度必须在 (0,1]，且步数乘强度至少为 1。')
    if method=='quality' and intent.action=='replace':
        from core.local_replace import replace,replace_at_reference_scale
        pipe=get_pipe('inpaint')
        result,alpha,details=replace(image,bundle,intent.new_content,pipe,
            torch,device,steps,guidance,strength,seed,dilate,feather)
        from core.result_check import check_replacement
        check=check_replacement(result,bundle,intent.new_content)
        passes=details['instance_count']
        attempts=[{'round':0,'seed':int(seed),'guidance':float(guidance),'check':check,'accepted':True}]
        candidates=[result]
        # Retry only missed regions, with a fixed bound and reproducible seeds.
        # Preserve a candidate only if it confirms more of the selected regions.
        for attempt in range(1,3):
            if check.get('status')!='needs_review':break
            missing=[i-1 for i in check['unconfirmed_regions']]
            retry_bundle={'components':[bundle['components'][i] for i in missing]}
            reference_retry=attempt==2 and max(image.size)>512
            retry_seed=int(seed) if reference_retry else int(seed)+1009*attempt
            retry_guidance=float(guidance) if reference_retry else max(6.,float(guidance)-1.5*attempt)
            generate_retry=replace_at_reference_scale if reference_retry else replace
            candidate,candidate_alpha,retry_details=generate_retry(image,retry_bundle,intent.new_content,pipe,
                torch,device,steps,retry_guidance,strength,retry_seed,dilate,feather,base_result=result)
            candidate_check=check_replacement(candidate,bundle,intent.new_content)
            passes+=len(missing)
            accepted=candidate_check.get('confirmed_regions',-1)>check.get('confirmed_regions',-1)
            attempts.append({'round':attempt,'seed':retry_seed,'guidance':retry_guidance,
                'strategy':'reference_scale_crop' if reference_retry else 'native_crop_retry',
                'original_regions':[i+1 for i in missing],'generation':retry_details,
                'check':candidate_check,'accepted':accepted})
            candidates.append(candidate)
            if accepted:
                result=candidate;alpha=np.maximum(alpha,candidate_alpha);check=candidate_check
        details.update(automatic_check=check,attempts=attempts,generated_instance_passes=passes)
        effective=(alpha>0).astype(np.uint8)*255
        if device=='cuda':torch.cuda.synchronize()
        return {'input':image,'result':result,'raw':result,'mask':effective,
            'mask_overlay':masker.overlay(image,effective),'bundle':dict(bundle,hard=effective,soft=alpha),
            'intent':intent,'strength':strength,'time':time.perf_counter()-t0,
            'peak_vram_mb':torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
            'device':device,'engine':'sd_inpaint_local_instances','engine_info':details,
            'effective_steps':int(steps)*passes,'raw_kind':'local_instance_composites',
            'candidates':candidates if len(candidates)>1 else []}
    def generate(source, mask, run_seed,text_prompt=None):
        extra={'padding_mask_crop':32} if method=='local_crop' else {}
        return get_pipe('inpaint')(
            prompt=text_prompt or intent.edit_prompt, negative_prompt=intent.negative_prompt,
            image=source, mask_image=Image.fromarray(mask), width=source.width, height=source.height,
            num_inference_steps=int(steps), guidance_scale=float(guidance), strength=strength,
            generator=torch.Generator(device).manual_seed(int(run_seed)),**extra).images[0]
    instance_count=0
    if method in {'ip2p','ip2p_mask'}:
        raw = get_pipe('ip2p')(prompt=_to_instruction(intent), image=image,
            num_inference_steps=int(steps), guidance_scale=float(guidance), image_guidance_scale=1.5,
            generator=torch.Generator(device).manual_seed(int(seed))).images[0]
        raw = raw.resize(image.size, Image.Resampling.LANCZOS)
        result = blend(image,raw,soft) if method == 'ip2p_mask' else raw
    elif method == 'ours_instances':
        if len(bundle['components']) > 8:
            raise ValueError('连通域超过 8 个，请先修正碎片掩码。')
        result = image
        singular={'dogs':'dog','cats':'cat','tigers':'tiger','zebras':'zebra','cows':'cow','horses':'horse',
                  'vases':'vase','bowls':'bowl','armchairs':'armchair','benches':'bench'}
        instance_prompt=re.sub(r'\b('+ '|'.join(singular)+r')\b',lambda m:singular[m.group()],intent.edit_prompt)
        # Components split BEFORE dilation. Touching instances may still be inseparable.
        for i,component in enumerate(bundle['components']):
            h,s = masker.refine_binary(component, 0 if bundle['source']=='manual' else dilate, feather)
            generated = generate(result,h,int(seed)+i,instance_prompt)
            result = blend(result,generated,s)
            instance_count+=1
        raw = result  # sequential composites, not an unblended single generation
    else:
        raw = generate(image,hard,seed)
        result = raw if method == 'inpaint_raw' else blend(image,raw,
                  (hard>127).astype(np.float32) if method=='inpaint_hard' else soft)
    if device == 'cuda':
        torch.cuda.synchronize()
    return {'input': image, 'result': result, 'raw': raw, 'mask': hard,
            'mask_overlay': masker.overlay(image,hard) if hard is not None else None,
            'bundle': bundle, 'intent': intent, 'strength': strength,
            'time': time.perf_counter()-t0,
            'peak_vram_mb': torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
            'device': device,'engine':'instruct_pix2pix' if method in {'ip2p','ip2p_mask'} else 'sd_inpaint',
            'effective_steps':int(steps)*max(1,instance_count),
            'engine_info':{'instance_count':instance_count,'steps_per_instance':int(steps)} if instance_count else None,
            'raw_kind':'sequential_composites' if instance_count else 'diffusion_output'}

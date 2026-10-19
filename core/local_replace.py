"""Replace individual selected objects in enlarged crops; retain native canvas."""
import re
import numpy as np
from PIL import Image
from core import masker


def replacement_prompt(content):
    # Singular nouns are required for each individual crop, not a full-scene count.
    singular={'tigers':'tiger','zebras':'zebra','cows':'cow','dogs':'dog','cats':'cat',
              'armchairs':'armchair','chairs':'chair','bowls':'bowl','benches':'bench'}
    noun=re.sub(r'^(?:a|an|the)\s+','',content.lower().strip())
    noun=singular.get(noun,noun)
    detail={'tiger':'tiger with orange fur and black stripes',
            'zebra':'zebra with black and white stripes',
            'cow':'brown and white cow with four legs',
            'armchair':'armchair with a seat, backrest and armrests',
            'bowl':'ceramic bowl with a visible open rim',
            'bench':'bench with a seat and supporting legs'}.get(noun,'detailed '+noun)
    return f'a photograph of one {noun}, {detail}, realistic, detailed, natural lighting'


def replace(image,bundle,content,pipe,torch,device,steps,guidance,strength,seed,dilate,feather,base_result=None):
    components=bundle['components']
    if not 1<=len(components)<=8:
        raise ValueError('局部替换需要 1 至 8 个对象选区，请先修正选区。')
    prompt=replacement_prompt(content)
    negative='blurry, deformed, malformed, missing object, empty background, extra limbs, watermark, text'
    base=image if base_result is None else base_result
    result=np.asarray(base,dtype=np.float32).copy()
    combined_alpha=np.zeros((image.height,image.width),np.float32)
    details=[]
    for index,component in enumerate(components):
        hard,soft=masker.refine_binary(component,dilate,feather)
        mask=Image.fromarray(hard)
        # A square processing canvas gives each object the model's full 512px resolution.
        coords=pipe.image_processor.get_crop_region(mask,512,512,pad=32)
        cropped=image.crop(coords).resize((512,512),Image.Resampling.LANCZOS)
        crop_mask=mask.crop(coords).resize((512,512),Image.Resampling.NEAREST)
        generated=pipe(prompt=prompt,negative_prompt=negative,image=cropped,mask_image=crop_mask,
            width=512,height=512,strength=strength,num_inference_steps=int(steps),
            guidance_scale=float(guidance),generator=torch.Generator(device).manual_seed(int(seed)+index)).images[0]
        canvas=image.copy()
        canvas.paste(generated.resize((coords[2]-coords[0],coords[3]-coords[1]),Image.Resampling.LANCZOS),coords[:2])
        alpha=soft[...,None]
        result=result*(1-alpha)+np.asarray(canvas,dtype=np.float32)*alpha
        combined_alpha=1-(1-combined_alpha)*(1-soft)
        details.append({'crop_box':list(coords),'seed':int(seed)+index,'processing_size':[512,512]})
    result=np.clip(np.rint(result),0,255).astype(np.uint8)
    result[combined_alpha==0]=np.asarray(base)[combined_alpha==0]
    return Image.fromarray(result),combined_alpha,{
        'instance_count':len(components),'steps_per_instance':int(steps),'prompt':prompt,
        'negative_prompt':negative,'crops':details,'native_canvas':True}


def replace_at_reference_scale(image,bundle,content,pipe,torch,device,steps,guidance,strength,seed,dilate,feather,base_result=None):
    """Try normalized crop geometry while projecting only edits onto the native canvas."""
    import cv2
    scale=min(1.,512/max(image.size))
    size=(max(8,int(image.width*scale)//8*8),max(8,int(image.height*scale)//8*8))
    working=image.resize(size,Image.Resampling.LANCZOS)
    components=[(np.asarray(Image.fromarray((np.asarray(c)>0).astype(np.uint8)*255).resize(
        size,Image.Resampling.NEAREST))>127).astype(np.uint8) for c in bundle['components']]
    # Objects too small to retain a pixel at the reference scale cannot be edited here.
    if any(not c.any() for c in components):
        return replace(image,bundle,content,pipe,torch,device,steps,guidance,strength,seed,dilate,feather,base_result)
    base=image if base_result is None else base_result
    edited,alpha,details=replace(working,{'components':components},content,pipe,torch,device,
        steps,guidance,strength,seed,dilate,feather,base.resize(size,Image.Resampling.LANCZOS))
    native_alpha=cv2.resize(alpha,image.size,interpolation=cv2.INTER_LINEAR)
    projected=np.asarray(edited.resize(image.size,Image.Resampling.LANCZOS))
    result=np.asarray(base).copy();support=native_alpha>0
    result[support]=projected[support]
    details.update(reference_size=list(size),native_projection=True)
    return Image.fromarray(result),native_alpha,details

"""Color-only editing that does not resynthesize geometry or texture.

This is an explicitly labeled chroma transform, not a diffusion model. Midtone
L* is retained; extreme lightness is compressed so white/black can gain chroma.
"""
import cv2
import numpy as np
from PIL import Image

COLORS = {
    'blue':(45,95,210), 'red':(205,45,45), 'green':(45,150,65),
    'yellow':(225,190,35), 'orange':(225,120,30), 'purple':(140,65,190),
    'pink':(215,100,155), 'brown':(130,80,45), 'gray':(128,128,128),
    'black':(30,30,30), 'white':(230,230,230), 'golden':(200,155,40),
}

def recolor(image, binary_mask, color):
    if color.lower() not in COLORS:
        raise ValueError('保纹理改色暂不支持该颜色，请选择扩散重绘方式。')
    source=np.asarray(image.convert('RGB'))
    region=np.asarray(binary_mask)>127
    if region.shape!=source.shape[:2] or not region.any():
        raise ValueError('改色掩码为空或与原图尺寸不符。')
    rgb=source.astype(np.float32)/255.
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    target=np.array([[COLORS[color.lower()]]],dtype=np.float32)/255.
    target_lab=cv2.cvtColor(target,cv2.COLOR_RGB2LAB)[0,0]
    alpha=region.astype(np.float32)
    # HSV saturation is unstable near black; do not infer hue from dark noise.
    colored=region & (hsv[...,1]>.35) & (hsv[...,2]>.2)
    gate_used=False
    # Only infer a dominant source hue when chromatic pixels actually dominate.
    # This preserves white stripes and dark trim on a saturated red/blue garment.
    if colored.sum()>=region.sum()*.4:
        angles=hsv[...,0][colored]
        weights=hsv[...,1][colored]
        hist,_=np.histogram(angles,bins=36,range=(0,360),weights=weights)
        smoothed=hist+np.roll(hist,1)+np.roll(hist,-1)
        index=int(np.argmax(smoothed));center=(index+.5)*10
        distance=np.abs((hsv[...,0]-center+180)%360-180)
        cluster=colored & (distance<30)
        if cluster.sum()>=colored.sum()*.55:
            saturation=float(np.median(hsv[...,1][cluster]))
            hue_weight=np.clip((40-distance)/15,0,1)
            saturation_weight=np.clip((hsv[...,1]-max(.12,saturation*.45))/.15,0,1)
            alpha*=hue_weight*saturation_weight
            gate_used=True
    proposed=lab.copy()
    if color.lower() in {'black','white'}:
        # White/black must change lightness: keep the original local contrast.
        selected=lab[...,0][region]
        center=float(np.median(selected))
        desired=24. if color.lower()=='black' else 84.
        proposed[...,0]=np.clip(lab[...,0]-center+desired,2,98)
        proposed[...,1:]*=.10
    else:
        # Pure white/black cannot acquire visible chroma at identical lightness.
        # Compress only the extremes monotonically, retaining local contrast.
        lightness=lab[...,0]
        proposed[...,0]=lightness-.5*np.maximum(lightness-70,0)+.5*np.maximum(12-lightness,0)
        source_chroma=np.linalg.norm(lab[...,1:],axis=-1)
        median=float(np.median(source_chroma[region]))
        variation=np.clip(source_chroma/max(median,8),.55,1.15) if gate_used else np.ones(region.shape,np.float32)
        light_taper=np.sqrt(np.clip(np.sin(proposed[...,0]*np.pi/100),0,1))
        proposed[...,1]=target_lab[1]*variation*light_taper
        proposed[...,2]=target_lab[2]*variation*light_taper
    converted=cv2.cvtColor(proposed,cv2.COLOR_LAB2RGB)
    # Finite blur confined to the selected region: never modify outside it.
    edge=cv2.GaussianBlur(region.astype(np.float32),(3,3),0)*region
    alpha*=edge
    result=np.rint(np.clip(rgb*(1-alpha[...,None])+converted*alpha[...,None],0,1)*255).astype(np.uint8)
    # Avoid float conversion roundoff for fully protected pixels.
    result[alpha==0]=source[alpha==0]
    return Image.fromarray(result),alpha,{'engine':'lab_chroma', 'source_hue_gate':gate_used,
        'color':color.lower(),'lightness_strategy':'retain midtone lightness; compress extremes for gamut' if color.lower() not in {'black','white'} else 'shift lightness, retain local contrast',
        'note':'A deterministic color transform; segmentation errors can still affect the wrong object.'}

def diagnostics(original,edited,mask,color):
    """Development diagnostics, not semantic success or segmentation accuracy."""
    roi=np.asarray(mask)>127
    if not roi.any():return {}
    a=cv2.cvtColor(np.asarray(original).astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,0]
    b=cv2.cvtColor(np.asarray(edited).astype(np.float32)/255,cv2.COLOR_RGB2LAB)[...,0]
    x,y=a[roi],b[roi]
    correlation=float(np.corrcoef(x,y)[0,1]) if x.std()>1e-6 and y.std()>1e-6 else None
    result={'lightness_mae':float(np.mean(np.abs(x-y))),
            'lightness_correlation':correlation}
    if color in COLORS:
        hsv=cv2.cvtColor(np.asarray(edited).astype(np.float32)/255,cv2.COLOR_RGB2HSV)
        target=cv2.cvtColor(np.array([[COLORS[color]]],np.float32)/255,cv2.COLOR_RGB2HSV)[0,0]
        if color in {'black','white','gray'}:
            match=(hsv[...,1]<.2)
            if color=='white':match &= hsv[...,2]>.75
            elif color=='black':match &= hsv[...,2]<.3
            else:match &= (hsv[...,2]>=.3)&(hsv[...,2]<=.8)
        else:
            distance=np.abs((hsv[...,0]-target[0]+180)%360-180)
            match=(distance<25)&(hsv[...,1]>.25)
        result['target_color_fraction_diagnostic']=float(match[roi].mean())
    return result

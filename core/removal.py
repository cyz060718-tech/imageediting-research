"""CPU LaMa background repair for object removal; no diffusion resynthesis."""
import hashlib
import warnings
import numpy as np
from PIL import Image
import config

_model=None


def load():
    global _model
    if _model is None:
        import torch
        path=config.LAMA_MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError('缺少 LaMa 删除模型，请运行 tools/download_lama.py，或选择 SD 重绘方式。')
        if hashlib.md5(path.read_bytes()).hexdigest()!=config.LAMA_MODEL_MD5:
            raise ValueError('LaMa 模型校验失败，请重新下载。')
        torch.set_num_threads(min(8,torch.get_num_threads()))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=FutureWarning,message='.*torch.jit.load.*')
            _model=torch.jit.load(str(path),map_location='cpu').eval()
    return _model


def remove(image,hard,soft):
    import torch
    original=np.asarray(image.convert('RGB'))
    mask=(np.asarray(hard)>127).astype(np.float32)
    alpha=np.asarray(soft,dtype=np.float32)
    if mask.shape!=original.shape[:2] or alpha.shape!=mask.shape or not mask.any():
        raise ValueError('删除选区为空或与原图尺寸不一致。')
    height,width=mask.shape
    pad_h=max(64,((height+7)//8)*8)-height
    pad_w=max(64,((width+7)//8)*8)-width
    rgb=np.pad(original.astype(np.float32)/255,((0,pad_h),(0,pad_w),(0,0)),mode='reflect')
    padded_mask=np.pad(mask,((0,pad_h),(0,pad_w)),mode='constant')
    source=torch.from_numpy(rgb.transpose(2,0,1).copy())[None]
    selected=torch.from_numpy(padded_mask.copy())[None,None]
    with torch.inference_mode():
        output=load()(source,selected)[0,:,:height,:width].permute(1,2,0).numpy()
    raw=np.clip(np.rint(output*255),0,255).astype(np.uint8)
    result=np.clip(np.rint(original.astype(np.float32)*(1-alpha[...,None])+raw*alpha[...,None]),0,255).astype(np.uint8)
    result[alpha==0]=original[alpha==0]
    return Image.fromarray(result),Image.fromarray(raw)

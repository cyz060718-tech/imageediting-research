"""Independent GT evaluation. Missing reference means unavailable, never a predicted ROI."""
import numpy as np
from PIL import Image, ImageFilter
import config

_cache = {}

def region_metrics(original, edited, region):
    a, b = np.asarray(original, dtype=np.float64), np.asarray(edited, dtype=np.float64)
    if a.shape != b.shape or a.shape[:2] != region.shape:
        raise ValueError('图像与评价区域必须具有相同尺寸。')
    selected = (a-b)[np.asarray(region, dtype=bool)]
    if not len(selected):
        return {'mae': None, 'psnr': None, 'changed_fraction': None, 'exact': None}
    mse = float(np.mean(selected**2))
    return {'mae': float(np.mean(np.abs(selected))),
            'psnr': float(10*np.log10(255**2/mse)) if mse else float('inf'),
            'changed_fraction': float(np.mean(np.any(selected != 0, axis=1))), 'exact': mse == 0}

def clip_score(image, text):
    import torch
    if 'clip' not in _cache:
        from transformers import CLIPModel, CLIPProcessor
        path = config.model_path('clip')
        # Keep metrics on CPU to avoid competing with diffusion VRAM.
        _cache['clip'] = (CLIPProcessor.from_pretrained(path, local_files_only=True),
                          CLIPModel.from_pretrained(path, local_files_only=True).eval())
    processor, model = _cache['clip']
    with torch.inference_mode():
        out = model(**processor(text=[text], images=[image], return_tensors='pt', padding=True))
        a, b = out.image_embeds, out.text_embeds
        return float(100 * ((a/a.norm(dim=-1, keepdim=True)) * (b/b.norm(dim=-1, keepdim=True))).sum().clamp(min=0))

def spatial_lpips(original, edited, region):
    """LPIPS spatial map weighted by GT region; receptive fields can cross ROI boundaries."""
    import torch
    import torch.nn.functional as F
    if not np.any(region):
        return None
    if 'lpips' not in _cache:
        import lpips
        _cache['lpips'] = lpips.LPIPS(net='alex', spatial=True, verbose=False).eval()
    def tensor(im):
        return torch.from_numpy(np.array(im, dtype=np.float32).copy()).permute(2,0,1)[None] / 127.5 - 1
    with torch.inference_mode():
        distance = _cache['lpips'](tensor(original), tensor(edited))
        weights = F.interpolate(torch.from_numpy(region.astype(np.float32))[None,None],
                                size=distance.shape[-2:], mode='nearest')
        if not weights.sum():
            return None
        return float((distance * weights).sum()/weights.sum())

def evaluate(original, edited, target_text='', pred_mask=None, gt_mask=None, learned=False):
    from core.masker import mask_iou
    result = {'evaluation_region': 'human_gt' if gt_mask is not None else 'unavailable'}
    result.update({'global_'+k: v for k,v in region_metrics(original, edited,
                  np.ones((original.height, original.width), dtype=bool)).items()})
    if gt_mask is not None:
        gt = np.asarray(gt_mask) > 127
        background = ~gt
        ring = np.asarray(Image.fromarray(gt.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(11))) > 127
        ring &= ~gt
        for name, roi in [('background', background), ('boundary', ring)]:
            result.update({name+'_'+k: v for k,v in region_metrics(original, edited, roi).items()})
        if pred_mask is not None:
            result['mask_iou'] = mask_iou(pred_mask, gt_mask)
        if learned:
            result['background_spatial_lpips'] = spatial_lpips(original, edited, background)
    if learned and target_text:
        result['clip_score_proxy'] = clip_score(edited, target_text)
    return result

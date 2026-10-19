"""Text segmentation with explicit failure, components BEFORE dilation, manual masks."""
import cv2
import numpy as np
from PIL import Image, ImageOps
import config

_cache = {}

def predict_prob(image, text):
    if not text.strip():
        raise ValueError('分割目标为空，请重新描述或提供人工编辑掩码。')
    import torch
    if 'clipseg' not in _cache:
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        path = config.model_path('clipseg')
        _cache['clipseg'] = (CLIPSegProcessor.from_pretrained(path, local_files_only=True),
                            CLIPSegForImageSegmentation.from_pretrained(path, local_files_only=True).eval())
    processor, model = _cache['clipseg']
    with torch.inference_mode():
        inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt')
        prob = model(**inputs).logits.sigmoid().squeeze().cpu().numpy()
    return cv2.resize(prob, image.size, interpolation=cv2.INTER_LINEAR)

def refine_binary(binary, dilate=9, feather=11):
    if dilate < 0 or feather < 0:
        raise ValueError('膨胀与羽化参数不能为负数。')
    hard = (np.asarray(binary) > 0).astype(np.uint8)
    if dilate > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dilate) | 1, int(dilate) | 1))
        hard = cv2.dilate(hard, k)
    soft = hard.astype(np.float32)
    if feather > 1:
        soft = cv2.GaussianBlur(soft, (int(feather) | 1, int(feather) | 1), 0)
    return hard * 255, np.clip(soft, 0, 1)

def build_masks(prob, threshold=.4, dilate=9, feather=11, min_area=16):
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2 or not np.isfinite(prob).all() or not 0 <= prob.min() <= prob.max() <= 1:
        raise ValueError('概率图必须是有限的二维 [0,1] 数组。')
    if not 0 < threshold < 1:
        raise ValueError('阈值必须介于 0 与 1 之间。')
    binary = (prob >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = [(labels == i).astype(np.uint8) for i in range(1, count)
                  if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not components:
        raise ValueError(f'未检测到可靠目标（最大响应 {prob.max():.3f}）。请预览掩码、调整阈值或上传人工掩码。')
    binary = np.maximum.reduce(components)
    hard, soft = refine_binary(binary, dilate, feather)
    area = float((hard > 0).mean())
    if area > .85:
        raise ValueError(f'掩码覆盖 {area:.1%} 图像，超出局部编辑范围，请修正掩码。')
    return {'hard': hard, 'soft': soft, 'raw': binary * 255, 'prob': prob,
            'components': components, 'area_fraction': area, 'component_count': len(components),
            'source': 'clipseg'}

def validate_manual_mask(mask, source=None):
    """Reject photos rather than silently thresholding their bright background."""
    mask = ImageOps.exif_transpose(mask).convert('RGB')
    pixels = np.asarray(mask)
    if source is not None:
        source = ImageOps.exif_transpose(source).convert('RGB')
        if mask.size != source.size:
            raise ValueError('人工掩码须与原图同尺寸。')
        if np.array_equal(np.asarray(mask.convert('L')), np.asarray(source.convert('L'))):
            raise ValueError('人工选区中放入了原图。请切回自动识别，或上传白色选中、黑色保留的黑白掩码。')
    monochrome = pixels.max(axis=2).astype(np.int16)-pixels.min(axis=2).astype(np.int16) <= 8
    gray = np.asarray(mask.convert('L'))
    near_binary = (gray <= 10) | (gray >= 245)
    if monochrome.mean() < .998 or near_binary.mean() < .98:
        raise ValueError('这不是黑白选区图。请勿在人工选区中上传原照片；可使用自动识别，或手绘选区。白色编辑，黑色保留。')
    if not np.any(gray > 127):
        raise ValueError('人工掩码为空。白色表示要编辑的区域。')
    return mask.convert('L')

def manual_masks(mask, size, dilate=0, feather=11):
    mask = validate_manual_mask(mask)
    binary = np.asarray(mask.resize(size, Image.Resampling.NEAREST)) > 127
    if not binary.any():
        raise ValueError('人工掩码为空。白色表示要编辑的区域。')
    # Manual masks may cover large regions; their extent is explicitly chosen by user.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    comps = [(labels == i).astype(np.uint8) for i in range(1, count)]
    hard, soft = refine_binary(binary, dilate, feather)
    return {'hard': hard, 'soft': soft, 'raw': binary.astype(np.uint8) * 255, 'prob': None,
            'components': comps, 'area_fraction': float(binary.mean()),
            'component_count': len(comps), 'source': 'manual'}

def generate_bundle(image, text, **kwargs):
    return build_masks(predict_prob(image, text), **kwargs)

def overlay(image, mask, color=(255, 60, 60), alpha=.45):
    a = np.asarray(image.convert('RGB'), dtype=np.float32)
    a = np.where((mask > 127)[..., None], a * (1-alpha) + np.array(color) * alpha, a)
    return Image.fromarray(a.astype(np.uint8))

def mask_iou(pred, gt):
    p, g = np.asarray(pred) > 127, np.asarray(gt) > 127
    if p.shape != g.shape:
        raise ValueError('IoU 掩码尺寸不一致。')
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 1.

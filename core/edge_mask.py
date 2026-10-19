"""Optional image-guided boundary refinement, with conservative acceptance gates."""
import cv2
import numpy as np
from core import masker

def refine(image,bundle,dilate=15,feather=21):
    probability=bundle.get('prob')
    if probability is None:
        return bundle,{'accepted':False,'reason':'manual mask is kept unchanged'}
    binary=bundle['raw']>127
    peak=float(probability.max())
    labels=np.full(binary.shape,cv2.GC_PR_BGD,np.uint8)
    labels[binary]=cv2.GC_PR_FGD
    labels[probability<min(.10,peak*.2)]=cv2.GC_BGD
    sure_fg=(probability>=min(.70,peak*.85)) & binary
    if sure_fg.sum()<8 or not np.any(labels==cv2.GC_BGD):
        return bundle,{'accepted':False,'reason':'insufficient foreground/background seeds'}
    labels[sure_fg]=cv2.GC_FGD
    try:
        cv2.setRNGSeed(42)
        cv2.grabCut(np.asarray(image.convert('RGB')),labels,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
        selected=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD])
        ratio=float(selected.sum()/binary.sum())
        if not .45<=ratio<=1.8 or selected.mean()>.85:
            return bundle,{'accepted':False,'reason':'refinement changed area too much','area_ratio':ratio}
        # Component identification stays before morphological dilation.
        refined=masker.build_masks(selected.astype(np.float32),.5,dilate,feather)
        refined['prob']=probability
        refined['source']='clipseg_grabcut'
        return refined,{'accepted':True,'area_ratio':ratio,'engine':'grabcut'}
    except (ValueError,cv2.error) as exc:
        return bundle,{'accepted':False,'reason':str(exc)}

"""Conservative region/count diagnostics. These do not certify edit quality."""
import numpy as np


def match_regions(components,detections):
    regions=[]
    for component in components:
        ys,xs=np.nonzero(component)
        regions.append([int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)])
    pairs=[]
    for i,a in enumerate(regions):
        area_a=(a[2]-a[0])*(a[3]-a[1])
        for j,found in enumerate(detections):
            b=found['box'];area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            iou=intersection/max(area_a+area_b-intersection,1)
            if iou>=.15:pairs.append((iou,i,j))
    matched_regions=set();matched_detections=set()
    for _,i,j in sorted(pairs,reverse=True):
        if i not in matched_regions and j not in matched_detections:
            matched_regions.add(i);matched_detections.add(j)
    return {'selected_instances':len(regions),'confirmed_regions':len(matched_regions),
        'unconfirmed_regions':[i+1 for i in range(len(regions)) if i not in matched_regions],
        'status':'detected_in_selected_regions' if len(matched_regions)==len(regions) else 'needs_review',
        'note':'Automatic detection is a diagnostic, not proof of correct anatomy, identity or edit success.'}


def check_replacement(image,bundle,content):
    from core.grounded_mask import detect_boxes
    try:
        detections=detect_boxes(image,content)
        return {**match_regions(bundle['components'],detections),'query':content,'detections':detections}
    except Exception as exc:
        return {'status':'unavailable','note':'Automatic result check failed; inspect the image.','error':str(exc)}

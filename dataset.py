"""Dataset checks, planning sheet, and a test lock. No synthetic sample counts."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps
import config

DATA = config.ROOT / 'data'
MANIFEST = DATA / 'manifest.json'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def source_hashes():
    files=[config.ROOT/n for n in ['config.py','dataset.py','eval.py','bootstrap.py']]
    files+=list((config.ROOT/'core').glob('*.py'))
    return {str(p.relative_to(config.ROOT)):digest(p) for p in files}

def resolve(relative):
    path = (DATA / relative).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError(f'数据路径越界: {relative}')
    return path

def load_items():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))

def validate(items=None, split=None, require_gt=False):
    items = load_items() if items is None else items
    errors, warnings, seen_ids, pixels, groups, refs = [], [], set(), {}, {}, {}
    selected = [it for it in items if split is None or it.get('split') == split]
    if not selected:
        errors.append(f'没有已登记的 {split or "全部"} 样本。计划表中的空行不算数据。')
    for it in items:
        ident = it.get('id','<missing>')
        if not it.get('id') or ident in seen_ids:
            errors.append(f'ID 缺失或重复: {ident}')
        seen_ids.add(ident)
        if it.get('split') not in {'dev','test'} or it.get('action') not in {'color','remove','replace'}:
            errors.append(f'{ident}: split/action 无效')
        if not it.get('instruction','').strip() or not it.get('eval_text','').strip():
            errors.append(f'{ident}: instruction 和 eval_text 必填')
        try:
            path = resolve(it['image'])
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.load()
                size = im.size
                key = hashlib.sha256(str(size).encode()+im.tobytes()).hexdigest()
            group = it.get('source_group') or key
            for table,k in [(pixels,key),(groups,group)]:
                if k in table and table[k] != it['split']:
                    errors.append(f'{ident}: 同一原图/来源组跨 dev/test 泄漏')
                table[k] = it['split']
            refs[str(path)] = digest(path)
            for field in ['gt_mask','manual_mask']:
                if not it.get(field):
                    continue
                mp = resolve(it[field])
                with Image.open(mp) as mask:
                    mask = ImageOps.exif_transpose(mask).convert('L')
                    import numpy as np
                    values = np.asarray(mask)
                    if mask.size != size or not set(np.unique(values)).issubset({0,255}) or not (values>127).any():
                        errors.append(f'{ident}: {field} 须与原图同尺寸，且为非空 0/255 二值 PNG')
                refs[str(mp)] = digest(mp)
            if it in selected:
                if not it.get('gt_mask') or it.get('gt_reviewed') is not True:
                    (errors if require_gt else warnings).append(f'{ident}: 缺少经过人工审核的 GT，独立背景/IoU 指标不可用')
                if it.get('manual_mask') and it.get('gt_mask') and resolve(it['manual_mask']) == resolve(it['gt_mask']):
                    warnings.append(f'{ident}: oracle 输入使用 GT 本身，必须单独作为 oracle 结果报告')
        except (OSError,KeyError,ValueError) as exc:
            errors.append(f'{ident}: {exc}')
    return {'errors':list(dict.fromkeys(errors)), 'warnings':warnings,
            'registered':len(selected), 'by_action':dict(Counter(it.get('action') for it in selected)),
            'file_hashes':refs}

def freeze(settings_path):
    report = validate(split='test',require_gt=True)
    if report['errors']:
        raise ValueError('\n'.join(report['errors']))
    settings = json.loads(Path(settings_path).read_text(encoding='utf-8'))
    lock = {'manifest_sha256':digest(MANIFEST), 'files':report['file_hashes'], 'settings':settings,
            'source_hashes':source_hashes()}
    # Freeze is explicit; a subsequent change requires a new lock and a new test run.
    (DATA/'test_lock.json').write_text(json.dumps(lock,ensure_ascii=False,indent=2),encoding='utf-8')
    return lock

def check_lock(settings):
    path = DATA/'test_lock.json'
    if not path.exists():
        raise ValueError('正式测试尚未冻结。请先完成 GT 审核并执行 dataset.py freeze。')
    lock=json.loads(path.read_text(encoding='utf-8'))
    if lock['manifest_sha256'] != digest(MANIFEST) or lock['settings'] != settings:
        raise ValueError('数据清单或参数与冻结版本不一致，请明确建立新的测试版本。')
    if lock.get('source_hashes') != source_hashes():
        raise ValueError('实验代码与冻结版本不一致，请建立新版本。')
    for path,sha in lock['files'].items():
        if digest(path) != sha:
            raise ValueError(f'冻结后的数据发生变化: {path}')

def create_plan():
    path = DATA/'collection_plan.csv'
    if path.exists():
        raise FileExistsError(f'已有计划表，避免覆盖: {path}')
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        fields=['slot_id','split','action','difficulty','image','instruction','eval_text','gt_mask',
                'gt_reviewed','manual_mask','source_group','source_url','license','notes']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for split,counts in [('dev',{'color':7,'remove':7,'replace':6}),('test',{'color':40,'remove':40,'replace':40})]:
            for action,n in counts.items():
                for i in range(n):
                    w.writerow({'slot_id':f'{split}_{action}_{i+1:03d}','split':split,'action':action,
                                'difficulty':['simple','complex_background','small_target','multiple_instances'][i%4],
                                'gt_reviewed':'false'})
    return path

def import_plan():
    """Import only filled rows; keep the three original development samples."""
    items=load_items();ids={it['id'] for it in items}
    with (DATA/'collection_plan.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if not row['image'].strip():
                continue
            if row['slot_id'] in ids:
                raise ValueError(f"ID 已存在，直接编辑 manifest.json 更新: {row['slot_id']}")
            item={k:v.strip() for k,v in row.items() if v and k not in {'slot_id','gt_reviewed'}}
            item.update(id=row['slot_id'],gt_reviewed=row['gt_reviewed'].lower()=='true')
            items.append(item);ids.add(item['id'])
    report=validate(items)
    if report['errors']:
        raise ValueError('\n'.join(report['errors']))
    backup=DATA/'manifest.before_import.json'
    if backup.exists():
        raise FileExistsError('已有导入备份，请先检查上次导入并重命名备份。')
    backup.write_bytes(MANIFEST.read_bytes())
    MANIFEST.write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','plan','import-plan','freeze'])
    p.add_argument('--split',choices=['dev','test']);p.add_argument('--require-gt',action='store_true')
    p.add_argument('--settings',default=str(config.ROOT/'experiment.json'));a=p.parse_args()
    if a.command=='plan':print(create_plan());return
    if a.command=='import-plan':import_plan();return
    if a.command=='freeze':freeze(a.settings);print('测试数据与参数已冻结。');return
    report=validate(split=a.split,require_gt=a.require_gt)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(bool(report['errors']))

if __name__=='__main__':main()

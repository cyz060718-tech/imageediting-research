"""Download IOPaint's published LaMa checkpoint and verify its published digest."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json,urllib.request
import config

def main():
    path=config.LAMA_MODEL_PATH;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        pending=path.with_suffix('.download')
        urllib.request.urlretrieve(config.LAMA_MODEL_URL,pending)
        if hashlib.md5(pending.read_bytes()).hexdigest()!=config.LAMA_MODEL_MD5:
            raise ValueError('LaMa 下载校验失败，未启用该文件。')
        pending.replace(path)
    actual=hashlib.md5(path.read_bytes()).hexdigest()
    if actual!=config.LAMA_MODEL_MD5:raise ValueError('现有 LaMa 模型校验失败。请移开损坏文件后重试。')
    (path.parent/'provenance.json').write_text(json.dumps({'url':config.LAMA_MODEL_URL,'md5':actual,
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,
        'source':'https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py'},indent=2),encoding='utf-8')
    print('Verified:',path)

if __name__=='__main__':main()

"""Download only the optional detector/SAM weights; source images remain local."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor
import config

def download(pair):
    name,repo=pair
    print('Downloading',repo,flush=True)
    path=snapshot_download(repo,local_dir=config.EXTRA_MODEL_DIR/name,
        allow_patterns=['*.json','*.safetensors','vocab.txt'],max_workers=2)
    print('Ready',name,path,flush=True)

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,config.EXTRA_MODEL_REPOS.items()))

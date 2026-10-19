"""Portable project paths. Model weights and outputs are not tracked by Git."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'outputs'
MODEL_DIR = Path(os.getenv('IMAGE_EDIT_MODEL_DIR', ROOT / 'models'))
MODEL_REPOS = {
    'clipseg': 'CIDAS/clipseg-rd64-refined',
    'inpaint': 'stable-diffusion-v1-5/stable-diffusion-inpainting',
    'ip2p': 'timbrooks/instruct-pix2pix',
    'clip': 'openai/clip-vit-base-patch32',
}
EXTRA_MODEL_DIR = Path(os.getenv('IMAGE_EDIT_EXTRA_MODEL_DIR', ROOT / 'models_extra'))
LAMA_MODEL_PATH = EXTRA_MODEL_DIR / 'lama' / 'big-lama.pt'
LAMA_MODEL_URL = 'https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt'
LAMA_MODEL_MD5 = 'e3aa4aaa15225a33ec84f9f4bc47e500'
EXTRA_MODEL_REPOS = {'grounding_dino':'IDEA-Research/grounding-dino-tiny',
                     'sam':'facebook/sam-vit-base'}
# Explicit download_models.py is required to fetch missing weights.
def model_path(name):
    path = (EXTRA_MODEL_DIR if name in EXTRA_MODEL_REPOS else MODEL_DIR) / name
    if not path.is_dir() or not any(path.iterdir()):
        raise FileNotFoundError(f'缺少本地模型 {name}: {path}')
    return str(path)

IMAGE_SIZE = 512
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 9.0
DEFAULT_SEED = 42
MASK_THRESHOLD = 0.4
MASK_DILATE = 15
MASK_FEATHER = 21
DEFAULT_MASK_BACKEND = 'grounded_sam'
LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://api.deepseek.com')
LLM_MODEL = os.getenv('LLM_MODEL', 'deepseek-chat')

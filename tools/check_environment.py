"""Fail early on import/CUDA kernel errors, rather than trusting is_available alone."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap  # noqa: F401
import importlib.metadata
import json
import config

def main():
    result={'python':sys.version,'executable':sys.executable,'models':{}}
    result['optional_models']={name:(config.EXTRA_MODEL_DIR/name/'model.safetensors').is_file()
                               for name in config.EXTRA_MODEL_REPOS}
    try:
        import torch
        import torchvision
        result.update(torch=torch.__version__,torchvision=torchvision.__version__,
                      cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result['gpu']=torch.cuda.get_device_name(0)
            result['vram_gb']=round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)
            result['architectures']=torch.cuda.get_arch_list()
            x=torch.ones((16,16),device='cuda');result['cuda_kernel_test']=float((x@x).sum().cpu())
        for name in config.MODEL_REPOS:
            result['models'][name]=config.model_path(name)
        result['ok']=True
    except Exception as exc:
        result.update(ok=False,error=f'{type(exc).__name__}: {exc}')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['ok'] else 1)

if __name__=='__main__':main()

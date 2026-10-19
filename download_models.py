"""下载所需模型到 ./models/ 目录。

用法：
    python download_models.py            # 下载全部（约 5GB）
    python download_models.py clipseg clip   # 只下载指定模型

国内网络默认走 hf-mirror.com 镜像；如能直连 HuggingFace，设置环境变量 HF_ENDPOINT=https://huggingface.co
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

import config  # noqa: E402

SKIP = ("*.ckpt", "*.bin", "*.msgpack", "*.h5", "*.onnx", "*.pt", "*.pb", "*non_ema*", "*.md", "*.png", "*.jpg")


def download(name: str):
    repo = config.MODEL_REPOS[name]
    target = config.MODEL_DIR / name
    files = HfApi().list_repo_files(repo)
    ignore = list(SKIP)
    # 扩散模型只下 fp16 权重，省一半空间
    if any(f.endswith(".fp16.safetensors") for f in files):
        ignore += [f for f in files if f.endswith(".safetensors") and ".fp16." not in f]
    elif not any(f.endswith(".safetensors") for f in files):
        ignore.remove("*.bin")  # 仓库只有 .bin 权重
    print(f"==> 下载 {repo} -> {target}")
    snapshot_download(repo, local_dir=target, ignore_patterns=ignore)


if __name__ == "__main__":
    names = sys.argv[1:] or list(config.MODEL_REPOS)
    for n in names:
        download(n)
    print("全部完成。")

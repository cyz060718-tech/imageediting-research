"""Choose and verify a Windows PyTorch build without assuming one GPU model.

Selection is conservative; an actual tensor, convolution and torchvision NMS
probe decides whether CUDA works. Automatic selection falls back to CPU.
Official build pairs: https://pytorch.org/get-started/previous-versions/
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Build:
    channel: str
    torch: str = '2.11.0'
    torchvision: str = '0.26.0'

    @property
    def index(self):
        return 'https://download.pytorch.org/whl/' + self.channel


def version_tuple(value):
    return tuple(int(part) for part in value.strip().split('.'))


def select_build(gpus):
    """Rows are (driver version, compute capability) from nvidia-smi.

    Older GPU architectures use the official CUDA 11.8 / torch 2.7.1 pair.
    Driver thresholds are conservative Windows toolkit driver requirements,
    not a promise that every CUDA feature or every card will work.
    """
    if not gpus:
        return Build('cpu')
    try:
        driver = min(version_tuple(row[0]) for row in gpus)
        capability = min(version_tuple(row[1]) for row in gpus)
        newest = max(version_tuple(row[1]) for row in gpus)
    except (ValueError, IndexError):
        return Build('cpu')
    # A mixed set including Blackwell must not be given a pre-12.8 build.
    if newest >= (10, 0):
        if capability < (7, 5):
            return Build('cpu')
        if driver >= (580, 88):
            return Build('cu130')
        return Build('cu128') if driver >= (572, 61) else Build('cpu')
    if capability < (5, 0):
        return Build('cpu')
    if capability < (7, 5):
        return Build('cu118', '2.7.1', '0.22.1') if driver >= (522, 6) else Build('cpu')
    if driver >= (580, 88):
        return Build('cu130')
    if driver >= (560, 76):
        return Build('cu126')
    if driver >= (522, 6):
        return Build('cu118', '2.7.1', '0.22.1')
    return Build('cpu')


def detect_gpus():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version,compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20,
        )
        if result.returncode != 0:
            return []
        rows = [tuple(field.strip() for field in line.split(','))
                for line in result.stdout.splitlines() if line.strip()]
        return rows if all(len(row) == 2 for row in rows) else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def probe(device, require_cpu_build=False):
    # Always use a fresh process: importing torch in the installer process and
    # then replacing its DLLs would keep old libraries loaded on Windows.
    code = (
        "import bootstrap; import torch,torchvision; "
        f"device={device!r}; "
        f"assert not {require_cpu_build!r} or torch.version.cuda is None, 'CPU build required'; "
        "x=torch.ones((16,16),device=device); "
        "assert float((x@x).sum().cpu()) == 4096.0; "
        "dtype=torch.float16 if device=='cuda' else torch.float32; "
        "y=torch.nn.functional.conv2d(torch.ones((1,1,8,8),device=device,dtype=dtype),"
        "torch.ones((1,1,3,3),device=device,dtype=dtype)); "
        "assert float(y.sum().cpu()) == 324.0; "
        "boxes=torch.tensor([[0.,0.,4.,4.],[0.,0.,4.,4.]],device=device); "
        "keep=torchvision.ops.nms(boxes,torch.tensor([.9,.8],device=device),.5); "
        "assert keep.cpu().tolist()==[0]; "
        "print('Verified PyTorch',torch.__version__,'torchvision',torchvision.__version__,'on',device)"
    )
    try:
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
                                capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=120)
        if result.returncode == 0:
            print(result.stdout.strip(), flush=True)
            return True
        detail = (result.stderr or result.stdout).strip()
        print(f'{device.upper()} check failed: {detail[-2500:]}', flush=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'{device.upper()} check failed: {exc}', flush=True)
    return False


def install(build, index_override=None):
    packages = ['torch', 'torchvision'] if index_override else [
        'torch==' + build.torch, 'torchvision==' + build.torchvision]
    index = index_override or build.index
    print('Installing', ', '.join(packages), 'from', index, flush=True)
    command = [sys.executable, '-m', 'pip', 'install', '--upgrade', '--force-reinstall',
               '--only-binary=:all:', *packages, '--index-url', index,
               '--retries', '3', '--timeout', '60']
    return subprocess.run(command, cwd=ROOT).returncode == 0


def ensure_torch(mode='auto', index_override=None):
    gpus = [] if mode == 'cpu' else detect_gpus()
    build = Build('cpu') if mode == 'cpu' else select_build(gpus)
    if index_override:
        channel = 'cpu' if index_override.rstrip('/').lower().endswith('/cpu') else 'custom-cuda'
        if mode == 'cpu' and channel != 'cpu':
            raise RuntimeError('cpu mode requires a CPU TORCH_INDEX_URL.')
        build = Build(channel)
    print('Selected PyTorch channel:', build.channel, flush=True)
    device = 'cpu' if build.channel == 'cpu' else 'cuda'
    # The app chooses CUDA from torch.cuda.is_available(). Merely probing a
    # CUDA build on the CPU would not enforce our automatic CPU decision.
    require_cpu = device == 'cpu'
    if probe(device, require_cpu_build=require_cpu):
        return device
    if install(build, index_override) and probe(device, require_cpu_build=require_cpu):
        return device
    if device == 'cuda' and not index_override:
        print('WARNING: CUDA could not be installed or verified. Trying CPU mode.', flush=True)
        cpu = Build('cpu')
        if install(cpu) and probe('cpu', require_cpu_build=True):
            print('CPU fallback enabled. Image generation will be slower.', flush=True)
            return 'cpu'
    raise RuntimeError('No usable PyTorch build. See the errors above. Check network access '
                       'and the Microsoft Visual C++ x64 runtime. '
                       'https://aka.ms/vc14/vc_redist.x64.exe')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['auto', 'cpu'], default='auto')
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        print('ERROR: Run the project install BAT. This helper only modifies the project .venv.',
              file=sys.stderr, flush=True)
        return 1
    try:
        ensure_torch(args.mode, os.environ.get('TORCH_INDEX_URL'))
        return 0
    except (OSError, RuntimeError) as exc:
        print('ERROR:', exc, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

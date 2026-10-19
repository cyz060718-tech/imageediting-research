"""Download or import LaMa with TLS verification and an independent digest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import config


def digests(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def download_checked(url, pending, expected_md5, client, attempts=3, sleep=time.sleep):
    """Restart incomplete LaMa downloads; never publish unverified bytes."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            print(f'Downloading LaMa, attempt {attempt}/{attempts}', flush=True)
            with client.stream('GET', url) as response:
                response.raise_for_status()
                received, next_update = 0, 32 * 1024 * 1024
                with pending.open('wb') as stream:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        stream.write(chunk)
                        received += len(chunk)
                        if received >= next_update:
                            print(f'LaMa: {received // (1024 * 1024)} MiB received', flush=True)
                            next_update += 32 * 1024 * 1024
            if digests(pending)[0] != expected_md5:
                raise ValueError('LaMa checksum mismatch; downloaded bytes were not enabled.')
            return
        except (httpx.HTTPError, OSError, ValueError) as exc:
            last_error = exc
            print(f'LaMa attempt {attempt} failed: {type(exc).__name__}: {exc}', flush=True)
            if attempt < attempts:
                sleep(attempt)
    raise RuntimeError('LaMa download failed after retries. Check HTTPS proxy and certificate '
                       'settings, or use --source with a verified big-lama.pt file. '
                       'Other downloaded models can be reused.') from last_error


def prepare(path, url, expected_md5, source=None, client=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pending = path.with_suffix('.download')
        if source is not None:
            source = Path(source)
            if digests(source)[0] != expected_md5:
                raise ValueError('Source LaMa checksum mismatch. The file was not imported.')
            shutil.copyfile(source, pending)
        elif client is not None:
            download_checked(url, pending, expected_md5, client)
        else:
            # Use the same HTTP stack as Hugging Face Hub. Keep normal TLS
            # validation, proxy environment and certificate settings enabled.
            timeout = httpx.Timeout(60, connect=30)
            with httpx.Client(follow_redirects=True, timeout=timeout) as connection:
                download_checked(url, pending, expected_md5, connection)
        if digests(pending)[0] != expected_md5:
            raise ValueError('LaMa checksum mismatch before publishing the model.')
        pending.replace(path)
    actual, sha256 = digests(path)
    if actual != expected_md5:
        raise ValueError('Existing LaMa checksum mismatch. Back up that file and retry.')
    (path.parent / 'provenance.json').write_text(json.dumps({
        'url': url, 'md5': actual, 'sha256': sha256, 'bytes': path.stat().st_size,
        'source': 'https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py',
    }, indent=2), encoding='utf-8')
    print('Verified:', path, flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Import an existing big-lama.pt after checksum validation')
    args = parser.parse_args()
    try:
        prepare(config.LAMA_MODEL_PATH, config.LAMA_MODEL_URL, config.LAMA_MODEL_MD5, args.source)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print('ERROR:', exc, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Download public software/weights only. Never sends user videos anywhere."""
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVISION = '52d6c8ffea26cc873ac5ad116f8631268d7eb503'
REPO = 'Qwen/Qwen3-VL-2B-Instruct-GGUF'
MODEL_NAME = 'Qwen3VL-2B-Instruct-Q4_K_M.gguf'
PROJECTOR_NAME = 'mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf'
RELEASE = 'b11714'
ARCHIVE = f'llama-{RELEASE}-bin-win-cpu-x64.zip'
MODEL_DIR = ROOT / 'models' / 'qwen3-vl-2b'
ENGINE_DIR = ROOT / '.tools' / f'llama-{RELEASE}'
FILES = [
    (f'https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}/{ARCHIVE}',
     ROOT / '.tools' / ARCHIVE, 'f78eded047061a3f86372db426797400a2f1bb0ec37868d5927dd820eb3386c0'),
    (f'https://huggingface.co/{REPO}/resolve/{REVISION}/{MODEL_NAME}',
     MODEL_DIR / MODEL_NAME, '089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae'),
    (f'https://huggingface.co/{REPO}/resolve/{REVISION}/{PROJECTOR_NAME}',
     MODEL_DIR / PROJECTOR_NAME, 'f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82'),
]


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_verified(url: str, target: Path, digest: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256(target) == digest:
        print(f'Already verified: {target.name}', flush=True)
        return
    partial = target.with_suffix(target.suffix + '.part')
    try:
        print(f'Downloading: {target.name}', flush=True)
        with urllib.request.urlopen(url, timeout=60) as source, partial.open('wb') as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        if sha256(partial) != digest:
            raise ValueError(f'SHA256 mismatch: {target.name}')
        partial.replace(target)
        print(f'Verified: {target.name}', flush=True)
    finally:
        partial.unlink(missing_ok=True)


def main():
    for url, target, digest in FILES:
        download_verified(url, target, digest)
    # Only extract the publisher's pinned, checksum-verified archive.
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(FILES[0][1]) as archive:
        archive.extractall(ENGINE_DIR)
    print('Installed. Next: run start-model.ps1', flush=True)


if __name__ == '__main__':
    main()

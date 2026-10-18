"""Download and verify public model weights during build, never during inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.features import ENCODER_URL, ENCODER_PATH, ENCODER_SHA256
from app.model_setup import download_verified

if __name__ == '__main__':
    download_verified(ENCODER_URL, ENCODER_PATH, ENCODER_SHA256)

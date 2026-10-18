"""Frozen ResNet features; no network calls or dataset labels at inference time."""
import math
from pathlib import Path

import cv2
import numpy as np

from app.training import temporal_features

FEATURE_VERSION = 'resnet18-two-views-temporal-v1'
ENCODER_SHA256 = '4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c'
ENCODER_REVISION = '4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177'
ENCODER_URL = ('https://media.githubusercontent.com/media/onnx/models/' + ENCODER_REVISION +
               '/validated/vision/classification/resnet/model/resnet18-v1-7.onnx')
ROOT = Path(__file__).resolve().parent.parent
ENCODER_PATH = ROOT / 'models' / 'resnet18' / 'resnet18-v1-7.onnx'


def preprocess(image: np.ndarray, full_view: bool = False) -> np.ndarray:
    height, width = image.shape[:2]
    if full_view:
        scale = 224 / max(height, width)
        resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
        crop = np.full((224, 224, 3), [104, 116, 124], dtype=np.uint8)
        top, left = (224 - resized.shape[0]) // 2, (224 - resized.shape[1]) // 2
        crop[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    else:
        scale = 256 / min(height, width)
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        top, left = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
        crop = resized[top:top + 224, left:left + 224]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def load_onnx(path: Path):
    # The OpenCV filename overload fails on some Windows paths containing Cyrillic.
    return cv2.dnn.readNetFromONNX(np.frombuffer(path.read_bytes(), dtype=np.uint8))


class FrameEncoder:
    def __init__(self, path: Path = ENCODER_PATH):
        from app.model_setup import sha256
        if not path.is_file() or sha256(path) != ENCODER_SHA256:
            raise ValueError('Визуальный компонент не установлен или повреждён. Запустите setup-training.ps1.')
        cv2.setNumThreads(4)
        self.net = load_onnx(path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # OpenCV 4 prefixes imported ONNX nodes; OpenCV 5 uses the original names.
        layers = self.net.getLayerNames()
        self.pool_layer = next((name for name in layers if name.endswith('resnetv15_pool1_fwd')), None)
        if self.pool_layer is None:
            raise ValueError('Визуальный компонент не содержит ожидаемый слой признаков.')

    def encode(self, frame: np.ndarray) -> np.ndarray:
        views = []
        for full in (False, True):
            self.net.setInput(preprocess(frame, full_view=full))
            vector = self.net.forward(self.pool_layer).flatten().astype(np.float64)
            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError('Не удалось получить визуальные признаки.')
            views.append(vector / max(np.linalg.norm(vector), 1e-12))
        return np.concatenate(views)


def extract_video_features(path: Path, encoder: FrameEncoder) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError('Не удалось прочитать видео.')
        duration = count / fps
        if duration > 180:
            raise ValueError('Обученный прототип принимает записи до 3 минут.')
        indices = sorted({min(int(count) - 1, round(t * fps)) for t in np.arange(0, duration, 2)})
        if (count - 1 - indices[-1]) / fps > .5:
            indices.append(int(count) - 1)
        times, vectors = [], []
        end_adjustment = 0
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            adjusted = False
            if not ok and index >= int(count) - math.ceil(fps):
                # Some MOV headers count padded final frames. Search back at most one second.
                for offset in range(1, min(math.ceil(fps), index) + 1):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index - offset)
                    ok, frame = capture.read()
                    if ok:
                        end_adjustment = max(end_adjustment, int(count) - 1 - (index - offset))
                        adjusted = True
                        break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not ok or not math.isfinite(timestamp) or timestamp < 0 or timestamp > duration + .1:
                raise ValueError('Не удалось прочитать кадр и его время.')
            if adjusted and times and timestamp == times[-1]:
                continue
            if times and timestamp <= times[-1]:
                raise ValueError('Декодер вернул непоследовательные времена кадров.')
            times.append(timestamp)
            vectors.append(encoder.encode(frame))
        return {'times': np.asarray(times), 'features': temporal_features(np.asarray(vectors)).astype(np.float32),
                'duration_sec': np.asarray(duration), 'end_adjustment_frames': np.asarray(end_adjustment)}
    finally:
        capture.release()

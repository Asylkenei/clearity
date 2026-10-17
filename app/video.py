"""Подготовка видео. Здесь ещё нет модели и распознавания нарушений."""

import base64
import math
from pathlib import Path

import cv2


def prepare_video(path: Path) -> dict:
    """Читаем параметры и три обзорных кадра из локального файла."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Не удалось открыть видео. Проверьте, что файл не повреждён.")

        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Не удалось прочитать длительность и частоту кадров.")

        frames = []
        width = height = 0
        for fraction in (0.1, 0.5, 0.9):
            index = min(count - 1, int(count * fraction))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Не удалось прочитать один из обзорных кадров.")

            # Берём время декодера, а не относительное время уменьшенной копии.
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("Декодер не вернул временную метку кадра.")

            height, width = frame.shape[:2]
            if max(width, height) > 960:
                scale = 960 / max(width, height)
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("Не удалось сохранить кадр в JPEG.")
            frames.append({
                "timestamp_sec": round(timestamp, 3),
                "image_base64": base64.b64encode(jpeg.tobytes()).decode("ascii"),
            })

        return {
            "analysis_performed": False,
            "duration_sec": round(count / fps, 3),
            "duration_is_estimate": True,
            "fps": round(fps, 3),
            "frame_count": count,
            "width": width,
            "height": height,
            "frames": frames,
            "message": "Видео прочитано. Проверка нарушений пока не подключена.",
        }
    finally:
        capture.release()

"""Offline binary visual baseline. Does not infer an untrained semantic subtype."""
import base64
import time
from pathlib import Path

import cv2
import numpy as np

from app.features import FrameEncoder, extract_video_features
from app.training import WindowClassifier

ROOT = Path(__file__).resolve().parent.parent
SOURCE = {
    'title': 'Правила осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей',
    'document': 'Постановление Правления НБРК от 29.11.2019 № 231',
    'section': 'Глава 2. Порядок осуществления кассовых операций',
    'url': 'https://adilet.zan.kz/rus/docs/V1900019680',
    'relationship': 'context_only',
    'note': 'Общий нормативный контекст. Бинарная модель не устанавливает нарушение конкретного пункта.',
}


class TrainedDetector:
    def __init__(self, encoder=None, classifier=None):
        self.encoder = encoder or FrameEncoder()
        self.classifier = classifier or WindowClassifier(ROOT / 'deployment' / 'candidate.npz')
        cv2.setNumThreads(1)

    def analyze(self, path: Path, deadline: float) -> dict:
        started = time.monotonic()
        data = extract_video_features(path, self.encoder)
        if time.monotonic() >= deadline:
            raise ValueError('Истекло время анализа. Результат не сформирован.')
        scores = self.classifier.scores(data['features'])
        times = data['times']
        groups = []
        for index in np.flatnonzero(scores >= self.classifier.threshold):
            if groups and index == groups[-1][-1] + 1:
                groups[-1].append(int(index))
            else:
                groups.append([int(index)])
        violations = []
        capture = cv2.VideoCapture(str(path))
        try:
            for group in groups[:12]:
                peak = max(group, key=lambda i: scores[i])
                evidence = []
                for index in sorted(set([group[0], peak, group[-1]])):
                    capture.set(cv2.CAP_PROP_POS_MSEC, float(times[index]) * 1000)
                    ok, frame = capture.read()
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if float(times[index]) == 0 and -1.1 / capture.get(cv2.CAP_PROP_FPS) <= timestamp < 0:
                        timestamp = 0.0
                    if not ok:
                        raise ValueError('Не удалось прочитать кадр-доказательство.')
                    h, w = frame.shape[:2]
                    if max(h, w) > 960:
                        frame = cv2.resize(frame, (round(w * 960 / max(h, w)), round(h * 960 / max(h, w))))
                    ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise ValueError('Не удалось сохранить кадр-доказательство.')
                    evidence.append({'timestamp_sec': round(timestamp, 3),
                                     'image_base64': base64.b64encode(jpeg).decode('ascii')})
                violations.append({
                    'type': 'Визуальный признак возможного нарушения кассовой дисциплины',
                    'timecode_sec': round(float(times[peak]), 3),
                    'description': 'Визуальные признаки выбранного фрагмента похожи на размеченные примеры возможных нарушений. '
                                   'Нужен просмотр кадров человеком; конкретное действие и правовая квалификация моделью не установлены.',
                    'evidence_frames': evidence, 'score': round(float(scores[peak]), 4),
                    'normative_source': dict(SOURCE),
                })
        finally:
            capture.release()
        return {'verdict': 'violation' if violations else 'no_violation', 'violations': violations,
                'duration_sec': round(float(data['duration_sec']), 3),
                'elapsed_sec': round(time.monotonic() - started, 2),
                'sampled_frames': len(times), 'review_required': True,
                'classification_validated': False, 'normative_validated': False,
                'model': 'Frozen ResNet18 + temporal logistic regression',
                'message': 'Экспериментальный бинарный прогноз, а не юридическое заключение. '
                           'Подтипы событий не обучены. На 8 отложенных видео accuracy = 50%. '
                           'no_violation означает отсутствие сигнала модели, а не гарантию отсутствия нарушений.'}

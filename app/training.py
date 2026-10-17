"""Small local supervised classifier; clip labels are weak labels, not legal rules."""
import random
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def split_records(records: list[dict], seed: int = 17) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record['sha256']].append(record)
    classes = defaultdict(list)
    for digest, rows in sorted(grouped.items()):
        labels = {r['label'] for r in rows}
        if len(labels) != 1:
            raise ValueError('Одинаковые видео имеют разные метки; исправьте разметку вручную.')
        label = labels.pop()
        if label not in ('violation', 'no_violation'):
            raise ValueError('Неизвестная метка.')
        ordered = sorted(rows, key=lambda r: r['video_id'])
        times = sorted({float(r['label_timecode_sec']) for r in rows if r['label_timecode_sec'] is not None})
        if label == 'violation' and not times:
            raise ValueError('У нарушения отсутствует время события.')
        group = {'sha256': digest, 'video_ids': [r['video_id'] for r in ordered],
                 'path': ordered[0]['path'], 'label': label, 'positive_times': times}
        classes[label].append(group)
    if set(classes) != {'violation', 'no_violation'} or min(map(len, classes.values())) < 5:
        raise ValueError('Нужно хотя бы пять уникальных видео каждого класса.')
    rng = random.Random(seed)
    split = {'train': [], 'validation': [], 'test': []}
    for label in sorted(classes):
        groups = classes[label]
        rng.shuffle(groups)
        count = max(1, round(len(groups) * .2))
        split['test'].extend(groups[:count])
        split['validation'].extend(groups[count:2 * count])
        split['train'].extend(groups[2 * count:])
    for key in split:
        split[key].sort(key=lambda g: g['sha256'])
    return split


def training_indices(times: np.ndarray, group: dict) -> np.ndarray:
    if group['label'] == 'no_violation':
        return np.arange(len(times))
    # Two-second sampling grid: use only centres within one second of a marked event.
    mask = np.zeros(len(times), dtype=bool)
    for timestamp in group['positive_times']:
        distance = abs(times - timestamp)
        near = distance <= 1.01
        if not np.any(near):
            raise ValueError('Отметка события за пределами выбранных кадров; проверьте разметку.')
        mask |= near
    mask |= reviewed_window_mask(times, group)
    if not np.any(mask):
        raise ValueError('Для положительного видео нет обучающих окон.')
    return np.flatnonzero(mask)


def reviewed_window_mask(times: np.ndarray, group: dict) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    for interval in group.get('reviewed_intervals', []):
        start, end = interval['start_sec'], interval['end_sec']
        inside = (times >= start) & (times <= end)
        if not np.any(inside):
            midpoint = (start + end) / 2
            index = int(np.argmin(abs(times - midpoint)))
            # A short action may lie between sampled frames. Its temporal context must contain it.
            context_start, context_end = times[max(0, index - 1)], times[min(len(times) - 1, index + 1)]
            if abs(times[index] - midpoint) > 1.01 or not context_start <= midpoint <= context_end:
                raise ValueError('Нет подходящего временного окна для ручной отметки.')
            inside[index] = True
        mask |= inside
    return mask


def training_multipliers(times: np.ndarray, group: dict, indices: np.ndarray) -> np.ndarray:
    return np.where(reviewed_window_mask(times, group)[indices], 2., 1.)


def temporal_features(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError('Некорректные визуальные признаки.')
    features = []
    for i in range(len(vectors)):
        window = vectors[max(0, i - 1):min(len(vectors), i + 2)]
        change = window[-1] - window[0]
        motion = abs(np.diff(window, axis=0)).mean(axis=0) if len(window) > 1 else np.zeros(vectors.shape[1])
        features.append(np.concatenate([window.mean(axis=0), change, motion]))
    return np.asarray(features)


def fit_classifier(x, y, weights, c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, solver='lbfgs', random_state=17))
    model.fit(x, y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    return model


def save_classifier(model, path: Path, threshold: float):
    scale = model.named_steps['standardscaler']
    classifier = model.named_steps['logisticregression']
    path.parent.mkdir(parents=True, exist_ok=True)
    # Numeric arrays only. Loading never unpickles an arbitrary Python object.
    np.savez_compressed(path, mean=scale.mean_, scale=scale.scale_, coef=classifier.coef_[0],
                        intercept=classifier.intercept_[0], threshold=float(threshold))


class WindowClassifier:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.mean = data['mean']
            self.scale = data['scale']
            self.coef = data['coef']
            self.intercept = float(data['intercept'])
            self.threshold = float(data['threshold'])
        if (self.mean.ndim != 1 or self.mean.shape != self.scale.shape or self.mean.shape != self.coef.shape
                or not np.isfinite(np.concatenate([self.mean, self.scale, self.coef])).all()
                or np.any(self.scale <= 0) or not np.isfinite(self.intercept) or not 0 <= self.threshold <= 1):
            raise ValueError('Повреждены веса классификатора.')

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Неверная размерность признаков.')
        logits = ((x - self.mean) / self.scale) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def video_score(scores: np.ndarray) -> float:
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Не удалось получить оценки видео.')
    return float(np.max(scores))


def binary_metrics(labels, predictions) -> dict:
    y, p = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if y.shape != p.shape or not len(y):
        raise ValueError('Некорректные метки для оценки.')
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {'videos': len(y), 'true_positive': tp, 'true_negative': tn,
            'false_positive': fp, 'false_negative': fn, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'accuracy': (tp + tn) / len(y), 'balanced_accuracy': (recall + specificity) / 2}


def train_experiment(split: dict, features: dict, output: Path) -> dict:
    """Fit on train, choose C/threshold on validation, evaluate test only after choice."""
    from threadpoolctl import threadpool_limits
    output.mkdir(parents=True, exist_ok=True)
    xs, ys, ws = [], [], []
    counts = {label: sum(g['label'] == label for g in split['train'])
              for label in ('violation', 'no_violation')}
    for group in split['train']:
        data = features[group['sha256']]
        indices = training_indices(data['times'], group)
        x = data['features'][indices]
        multipliers = training_multipliers(data['times'], group, indices)
        xs.append(x)
        ys.extend([int(group['label'] == 'violation')] * len(x))
        per_video = len(split['train']) / (2 * counts[group['label']])
        ws.extend(per_video * multipliers / multipliers.sum())
    x, y, weights = np.concatenate(xs), np.asarray(ys), np.asarray(ws)

    def score_groups(model, groups):
        return [video_score(model.predict_proba(features[g['sha256']]['features'])[:, 1]) for g in groups]

    validation_labels = [int(g['label'] == 'violation') for g in split['validation']]
    best = None
    # Fixed before the test set is inspected; no test-driven search over seeds or models.
    with threadpool_limits(limits=4):
        for c in (.001, .01, .1, 1.0):
            model = fit_classifier(x, y, weights, c)
            scores = score_groups(model, split['validation'])
            for threshold in np.arange(.05, 1.001, .05):
                metrics = binary_metrics(validation_labels, np.asarray(scores) >= threshold)
                rank = (metrics['balanced_accuracy'], metrics['f1'], -c, float(threshold))
                if best is None or rank > best[0]:
                    best = (rank, model, c, float(threshold), metrics)
    _, model, c, threshold, validation_metrics = best
    save_classifier(model, output / 'candidate.npz', threshold)
    test_scores = score_groups(model, split['test'])
    test_labels = [int(g['label'] == 'violation') for g in split['test']]
    predictions = np.asarray(test_scores) >= threshold
    test_metrics = binary_metrics(test_labels, predictions)
    majority = int(counts['violation'] >= counts['no_violation'])
    baseline = binary_metrics(test_labels, [majority] * len(test_labels))
    test_rows = []
    near_mark = total_positive = 0
    for group, score, predicted in zip(split['test'], test_scores, predictions):
        data = features[group['sha256']]
        window_scores = model.predict_proba(data['features'])[:, 1]
        timestamp = float(data['times'][np.argmax(window_scores)])
        distance = None
        if group['label'] == 'violation':
            distance = min(abs(timestamp - t) for t in group['positive_times'])
            total_positive += 1
            near_mark += int(distance <= 4)
        test_rows.append({'video_ids': group['video_ids'], 'label': group['label'],
                          'predicted_positive': bool(predicted), 'score': score,
                          'peak_sec': timestamp, 'distance_to_mark_sec': distance})
    report = {'seed': 17, 'selected_c': c, 'threshold': threshold,
              'split_unique_videos': {k: len(v) for k, v in split.items()},
              'training_windows': len(x), 'training_positive_windows': int(y.sum()),
              'training_negative_windows': int((y == 0).sum()),
              'validation': validation_metrics, 'test': test_metrics, 'majority_baseline': baseline,
              'localization_diagnostic': {'positive_videos': total_positive, 'peak_within_4_sec': near_mark,
                  'note': 'Диагностика относительно одной/нескольких слабых отметок, не оценка всех событий.'},
              'test_predictions': test_rows, 'auto_promoted': False,
              'limitations': ['Всего 8 уникальных тестовых видео; оценка нестабильна.',
                  'Разделение по точному SHA256 не гарантирует независимость людей, камер и сессий.',
                  'Обучены только бинарные метки, без типов нарушения и нормативных правил.',
                  'ResNet заморожен, обучены scaler и линейный классификатор; веса Qwen не изменены.',
                  'Не измерено улучшение относительно Qwen на одинаковом тесте.'],
              'experiment_status': 'experimental_only'}
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report

"""Reproducible, offline-after-setup supervised experiment. No upload of user data."""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from app.features import (ENCODER_PATH, ENCODER_SHA256, ENCODER_URL, FEATURE_VERSION,
                          FrameEncoder, extract_video_features)
from app.model_setup import ROOT, download_verified, sha256
from app.training import split_records, train_experiment


def freeze_protocol(output: Path, protocol: dict):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Данные или протокол изменились. Нельзя менять условия уже начатого эксперимента.')
    else:
        path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT.parent.parent / 'work' / 'dataset' / 'manifest.json')
    parser.add_argument('--run-name', default='baseline-v1')
    parser.add_argument('--setup-only', action='store_true')
    args = parser.parse_args()
    if args.setup_only:
        download_verified(ENCODER_URL, ENCODER_PATH, ENCODER_SHA256)
        print('Training encoder installed; video data was not sent anywhere.', flush=True)
        return
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', args.run_name):
        raise ValueError('Имя эксперимента: латинские буквы, цифры, дефис или подчёркивание.')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    split = split_records(manifest['records'], seed=17)
    output = ROOT / 'models' / 'trained' / args.run_name
    protocol = {'seed': 17, 'feature_version': FEATURE_VERSION, 'encoder_sha256': ENCODER_SHA256,
                'sampling_sec': 2, 'window_radius_frames': 1, 'c_grid': [.001, .01, .1, 1.0],
                'threshold_grid': '0.05..1.0 step0.05', 'selection': 'validation_balanced_accuracy_then_f1',
                'splits': {k: [{x: g[x] for x in ('sha256', 'label', 'video_ids', 'positive_times')} for g in groups]
                           for k, groups in split.items()},
                'positive_training': 'centres within1.01s of provided point; other positive-video windows excluded',
                'auto_promotion': False}
    freeze_protocol(output, protocol)
    if (output / 'report.json').exists():
        print('Experiment already evaluated. Reusing the frozen report; test set will not be tuned.', flush=True)
        return
    (output / 'split-private.json').write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    cache = ROOT / 'runtime' / 'training' / FEATURE_VERSION
    cache.mkdir(parents=True, exist_ok=True)
    encoder = FrameEncoder()
    all_groups = [g for groups in split.values() for g in groups]
    features = {}
    started = time.monotonic()
    for number, group in enumerate(all_groups, 1):
        path = Path(group['path']).resolve()
        if not path.is_relative_to(args.manifest.resolve().parent) or not path.is_file():
            raise ValueError('Видео должно находиться в папке датасета.')
        if sha256(path) != group['sha256']:
            raise ValueError('Видео изменилось после создания каталога. Сначала обновите каталог.')
        cached = cache / (group['sha256'] + '.npz')
        if cached.exists():
            with np.load(cached, allow_pickle=False) as data:
                result = {key: data[key] for key in ('times', 'features', 'duration_sec')}
        else:
            result = extract_video_features(path, encoder)
            np.savez_compressed(cached, **result)
        if (result['features'].shape != (len(result['times']), 3072)
                or not np.isfinite(result['features']).all() or not np.isfinite(result['times']).all()):
            raise ValueError('Повреждён кэш признаков. Удалите только соответствующий файл кэша и повторите.')
        features[group['sha256']] = result
        print(json.dumps({'prepared_unique_videos': number, 'total': len(all_groups),
                          'elapsed_sec': round(time.monotonic() - started, 1)}), flush=True)
    report = train_experiment(split, features, output)
    metadata = {'feature_version': FEATURE_VERSION, 'encoder_sha256': ENCODER_SHA256,
                'sampling_sec': 2, 'feature_dim': 3072, 'threshold': report['threshold'],
                'run_name': args.run_name, 'review_required': True, 'normative_validated': False,
                'auto_promoted': False, 'training_labels': 'binary_clip_labels_plus_weak_point_timestamps'}
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('split_unique_videos', 'training_windows', 'selected_c', 'threshold',
                                          'validation', 'test', 'majority_baseline', 'localization_diagnostic',
                                          'auto_promoted')}, ensure_ascii=False), flush=True)
    print('Saved locally:', output, flush=True)


if __name__ == '__main__':
    main()

"""Read human annotations locally; keep historical validation/test groups unchanged."""
import copy
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path


def read_annotation_snapshot(path: Path) -> tuple[dict, set[str]]:
    if not path.is_file():
        raise ValueError('База ручной разметки не найдена. Сохраните наблюдения на местной странице /review.')
    try:
        # Read-only connection and one transaction: a concurrent UI save cannot mix snapshots.
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.execute('BEGIN')
            rows = db.execute('SELECT video_id, payload FROM annotations').fetchall()
            categories = {row[0] for row in db.execute('SELECT id FROM categories')}
        return {video_id: json.loads(payload) for video_id, payload in rows}, categories
    except (sqlite3.Error, ValueError) as exc:
        raise ValueError('Не удалось прочитать базу ручной разметки.') from exc


def prepare_reviewed_split(base_split: dict, records: list[dict], annotations: dict,
                           categories: set[str]) -> tuple[dict, dict]:
    if not annotations:
        raise ValueError('Нет сохранённых наблюдений. На местной странице /review нажмите «Сохранить наблюдение».')
    original = {r['video_id']: r for r in records}
    split_records = {video_id: g for groups in base_split.values() for g in groups for video_id in g['video_ids']}
    if set(original) != set(split_records) or any(
            original[vid]['sha256'] != group['sha256'] or original[vid]['label'] != group['label']
            for vid, group in split_records.items()):
        raise ValueError('Каталог отличается от базового разбиения. Нельзя незаметно менять проверочную выборку.')
    updated = copy.deepcopy(base_split)
    train = {vid: g for g in updated['train'] for vid in g['video_ids']}
    accepted, excluded = {}, []
    for video_id, annotation in sorted(annotations.items()):
        record = original.get(video_id)
        reason = None
        if record is None:
            reason = 'unknown_video'
        elif video_id not in train:
            reason = 'held_out_video'
        elif annotation.get('review_state') != 'reviewed':
            reason = 'not_reviewed'
        elif annotation.get('visibility') != 'clear':
            reason = 'not_clear'
        elif not str(annotation.get('observation') or '').strip():
            reason = 'missing_observation'
        else:
            start, end = annotation.get('event_start_sec'), annotation.get('event_end_sec')
            duration = record.get('duration_sec_estimate')
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                    or not math.isfinite(start) or not math.isfinite(end)
                    or start < 0 or end <= start or duration is None or end > duration):
                reason = 'invalid_interval'
            elif record['label'] == 'violation' and annotation.get('category_id') not in categories:
                reason = 'unknown_positive_category'
        if reason:
            excluded.append({'video_id': video_id, 'reason': reason})
            continue
        accepted[video_id] = annotation
        train[video_id].setdefault('reviewed_intervals', []).append({
            'video_id': video_id, 'start_sec': float(start), 'end_sec': float(end)})
    if not accepted:
        counts = dict(Counter(row['reason'] for row in excluded))
        raise ValueError('Нет сохранённых наблюдений, пригодных для обучения. Причины: ' + json.dumps(counts))
    for group in updated['train']:
        if group.get('reviewed_intervals') and group['label'] == 'violation':
            # Replace only this reviewed alias's weak point. Another duplicate's event is preserved.
            group['positive_times'] = sorted({float(original[vid]['label_timecode_sec'])
                                             for vid in group['video_ids'] if vid not in accepted
                                             and original[vid]['label_timecode_sec'] is not None})
    by_label = Counter(original[vid]['label'] for vid in accepted)
    audit = {'accepted_records': len(accepted),
             'accepted_unique_videos': len({original[vid]['sha256'] for vid in accepted}),
             'accepted_positive_records': by_label['violation'],
             'accepted_negative_records': by_label['no_violation'],
             'accepted_video_ids': sorted(accepted), 'excluded': excluded,
             'observation_text_used_as_features': False, 'reviewed_window_multiplier': 2,
             'historical_test_reused': True, 'validation_and_test_unchanged': True}
    return updated, audit

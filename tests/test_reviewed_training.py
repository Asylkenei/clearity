"""Human intervals must affect fitting without leaking validation/test annotations."""
import copy
import importlib.util
import json
import sqlite3
import sys

import numpy as np
import pytest

from app.training import training_indices, split_records


def reviewed():
    assert importlib.util.find_spec('app.reviewed_training'), 'Нужна загрузка ручной разметки'
    from app import reviewed_training
    return reviewed_training


def fixtures():
    records = [
        {'video_id': 'X-1', 'sha256': 'a', 'label': 'violation', 'label_timecode_sec': 5,
         'duration_sec_estimate': 20},
        {'video_id': 'X-2', 'sha256': 'a', 'label': 'violation', 'label_timecode_sec': 15,
         'duration_sec_estimate': 20},
        {'video_id': 'X-3', 'sha256': 'b', 'label': 'no_violation', 'label_timecode_sec': None,
         'duration_sec_estimate': 20},
        {'video_id': 'X-4', 'sha256': 'c', 'label': 'violation', 'label_timecode_sec': 5,
         'duration_sec_estimate': 20},
    ]
    split = {'train': [
        {'sha256': 'a', 'video_ids': ['X-1', 'X-2'], 'label': 'violation', 'positive_times': [5, 15]},
        {'sha256': 'b', 'video_ids': ['X-3'], 'label': 'no_violation', 'positive_times': []}],
        'validation': [{'sha256': 'c', 'video_ids': ['X-4'], 'label': 'violation', 'positive_times': [5]}],
        'test': []}
    annotation = {'review_state': 'reviewed', 'visibility': 'clear', 'observation': 'Synthetic visible action',
                  'category_id': 'cash_transfer', 'event_start_sec': 2., 'event_end_sec': 8.}
    return records, split, annotation


def test_reviewed_interval_replaces_its_source_point_but_keeps_other_duplicate_point():
    records, split, annotation = fixtures()
    before = copy.deepcopy(split)
    updated, audit = reviewed().prepare_reviewed_split(split, records, {'X-1': annotation},
                                                      {'cash_transfer'})
    assert split == before
    assert updated['train'][0]['positive_times'] == [15]
    assert updated['train'][0]['reviewed_intervals'] == [
        {'video_id': 'X-1', 'start_sec': 2., 'end_sec': 8.}]
    assert updated['validation'] == split['validation']
    assert audit['accepted_records'] == 1
    assert audit['accepted_unique_videos'] == 1


def test_empty_annotations_stop_instead_of_repeating_the_old_training():
    records, split, _ = fixtures()
    with pytest.raises(ValueError, match='Нет сохран'):
        reviewed().prepare_reviewed_split(split, records, {}, {'cash_transfer'})


def test_held_out_annotations_are_excluded_from_training():
    records, split, annotation = fixtures()
    updated, audit = reviewed().prepare_reviewed_split(
        split, records, {'X-1': annotation, 'X-4': annotation}, {'cash_transfer'})
    assert updated['validation'] == split['validation']
    assert audit['accepted_records'] == 1
    assert audit['excluded'][0]['reason'] == 'held_out_video'
    assert audit['excluded'][0]['video_id'] == 'X-4'


@pytest.mark.parametrize('change,reason', [
    ({'review_state': 'draft'}, 'not_reviewed'),
    ({'visibility': 'partial'}, 'not_clear'),
    ({'event_start_sec': None}, 'invalid_interval'),
    ({'event_start_sec': 9, 'event_end_sec': 3}, 'invalid_interval'),
    ({'event_end_sec': 21}, 'invalid_interval'),
    ({'observation': ''}, 'missing_observation'),
    ({'category_id': 'unknown'}, 'unknown_positive_category'),
])
def test_unusable_annotation_does_not_enter_training(change, reason):
    records, split, annotation = fixtures()
    negative = {**annotation, 'category_id': None}
    _, audit = reviewed().prepare_reviewed_split(
        split, records, {'X-1': {**annotation, **change}, 'X-3': negative}, {'cash_transfer'})
    assert audit['accepted_records'] == 1
    assert audit['excluded'][0]['reason'] == reason


def test_positive_training_uses_reviewed_boundaries_and_retains_unreviewed_duplicate_event():
    group = {'label': 'violation', 'positive_times': [15],
             'reviewed_intervals': [{'video_id': 'X-1', 'start_sec': 2., 'end_sec': 8.}]}
    times = np.arange(0., 20., 2.)
    assert training_indices(times, group).tolist() == [1, 2, 3, 4, 7, 8]


def test_narrow_interval_uses_surrounding_temporal_window_without_changing_boundaries():
    group = {'label': 'violation', 'positive_times': [],
             'reviewed_intervals': [{'video_id': 'X-1', 'start_sec': 2.8, 'end_sec': 3.2}]}
    assert training_indices(np.array([0., 2., 4., 6.]), group).tolist() == [1]
    assert group['reviewed_intervals'][0]['start_sec'] == 2.8


def test_reviewed_normal_action_gets_extra_weight_without_losing_other_negative_windows():
    from app import training
    assert hasattr(training, 'training_multipliers'), 'Нужны веса подтверждённых обычных действий'
    group = {'label': 'no_violation', 'positive_times': [],
             'reviewed_intervals': [{'video_id': 'X-3', 'start_sec': 2., 'end_sec': 4.}]}
    times = np.array([0., 2., 4., 6.])
    indices = training_indices(times, group)
    assert indices.tolist() == [0, 1, 2, 3]
    assert training.training_multipliers(times, group, indices).tolist() == [1., 2., 2., 1.]


def test_annotation_snapshot_reads_local_saved_rows_without_modifying_database(tmp_path):
    module = reviewed()
    assert hasattr(module, 'read_annotation_snapshot')
    path = tmp_path / 'annotations.sqlite3'
    _, _, annotation = fixtures()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE annotations (video_id TEXT, payload TEXT)')
        db.execute('CREATE TABLE categories (id TEXT)')
        db.execute('INSERT INTO annotations VALUES (?,?)', ('X-1', json.dumps(annotation)))
        db.execute('INSERT INTO categories VALUES (?)', ('cash_transfer',))
    before = path.read_bytes()
    data, categories = module.read_annotation_snapshot(path)
    assert data['X-1']['event_start_sec'] == 2
    assert categories == {'cash_transfer'}
    assert path.read_bytes() == before


def test_reviewed_cli_stops_before_creating_an_experiment_when_no_annotations_saved(tmp_path, monkeypatch):
    from app import train_run
    root = tmp_path / 'project'
    runtime = root / 'runtime'; runtime.mkdir(parents=True)
    with sqlite3.connect(runtime / 'annotations.sqlite3') as db:
        db.execute('CREATE TABLE annotations (video_id TEXT, payload TEXT)')
        db.execute('CREATE TABLE categories (id TEXT)')
    rows = [{'video_id': f'X-{i}', 'sha256': f'hash-{i}', 'label': 'violation' if i < 20 else 'no_violation',
             'label_timecode_sec': 5 if i < 20 else None, 'duration_sec_estimate': 20,
             'path': str(root / f'{i}.mov')} for i in range(40)]
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'records': rows}), encoding='utf-8')
    base = root / 'models' / 'trained' / 'baseline-v1'; base.mkdir(parents=True)
    (base / 'split-private.json').write_text(json.dumps(split_records(rows)), encoding='utf-8')
    (base / 'protocol.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(train_run, 'ROOT', root)
    monkeypatch.setattr(sys, 'argv', ['train_run', '--manifest', str(manifest), '--use-reviewed',
                                    '--base-run', 'baseline-v1', '--run-name', 'reviewed-v2'])
    with pytest.raises(ValueError, match='Нет сохран'):
        train_run.main()
    assert not (root / 'models' / 'trained' / 'reviewed-v2').exists()

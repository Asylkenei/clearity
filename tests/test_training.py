"""Protect split isolation, weak labels, real fitting and portable local predictions."""
import importlib.util

import numpy as np
import pytest


def training():
    assert importlib.util.find_spec('app.training'), 'Нужен воспроизводимый процесс обучения'
    from app import training
    return training


def records():
    rows = [{'video_id': f'X-{i}', 'sha256': f'hash-{i:03}', 'label': 'violation' if i < 20 else 'no_violation',
             'label_timecode_sec': 5 if i < 20 else None, 'path': f'not-used-{i}.mov'} for i in range(40)]
    rows.append({**rows[0], 'video_id': 'X-99', 'label_timecode_sec': 8})
    return rows


def test_duplicate_videos_stay_in_one_split_and_input_order_cannot_change_split():
    m = training()
    split = m.split_records(records(), seed=17)
    assert split == m.split_records(list(reversed(records())), seed=17)
    assert [len(split[k]) for k in ('train', 'validation', 'test')] == [24, 8, 8]
    hashes = [g['sha256'] for values in split.values() for g in values]
    assert len(hashes) == len(set(hashes)) == 40
    duplicate = next(g for values in split.values() for g in values if g['sha256'] == 'hash-000')
    assert duplicate['video_ids'] == ['X-0', 'X-99']
    assert duplicate['positive_times'] == [5, 8]


def test_conflicting_duplicate_labels_stop_training():
    rows = records()
    rows[-1]['label'] = 'no_violation'
    with pytest.raises(ValueError, match='разные метки'):
        training().split_records(rows)


def test_only_marked_positive_windows_are_used_and_unmarked_positive_video_is_not_negative():
    m = training()
    times = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    positive = {'label': 'violation', 'positive_times': [5]}
    assert m.training_indices(times, positive).tolist() == [2, 3]
    negative = {'label': 'no_violation', 'positive_times': []}
    assert m.training_indices(times, negative).tolist() == [0, 1, 2, 3, 4, 5]


def test_temporal_features_encode_change_in_order_and_reject_nan():
    m = training()
    vectors = np.array([[0., 1.], [1., 2.], [3., 4.]])
    out = m.temporal_features(vectors)
    assert out.shape == (3, 6)
    # Middle window: mean [4/3,7/3], signed change [3,3], mean absolute change [1.5,1.5].
    assert out[1] == pytest.approx([4/3, 7/3, 3, 3, 1.5, 1.5])
    with pytest.raises(ValueError):
        m.temporal_features(np.array([[float('nan'), 1.]]))


def test_saved_classifier_reproduces_real_fit_and_does_not_execute_pickle(tmp_path):
    m = training()
    x = np.array([[-3., 0.], [-2., 1.], [-1., 0.], [1., 0.], [2., 1.], [3., 0.]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = m.fit_classifier(x, y, np.ones(6), c=1.0)
    path = tmp_path / 'candidate.npz'
    m.save_classifier(model, path, threshold=0.5)
    loaded = m.WindowClassifier(path)
    assert loaded.scores(x) == pytest.approx(model.predict_proba(x)[:, 1], abs=1e-8)
    assert (loaded.scores(x) >= loaded.threshold).tolist() == y.astype(bool).tolist()


def test_evaluation_reports_misses_and_false_alarms():
    m = training()
    metrics = m.binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics['true_positive'] == 1
    assert metrics['false_negative'] == 1
    assert metrics['false_positive'] == 1
    assert metrics['true_negative'] == 1
    assert metrics['precision'] == 0.5
    assert metrics['recall'] == 0.5
    assert metrics['balanced_accuracy'] == 0.5


def test_video_max_aggregation_does_not_use_label_timecode():
    m = training()
    assert m.video_score(np.array([0.1, 0.8, 0.2])) == 0.8
    with pytest.raises(ValueError):
        m.video_score(np.array([]))


def test_test_set_cannot_influence_weights_or_threshold(tmp_path):
    m = training()
    assert hasattr(m, 'train_experiment'), 'Нужно обучение с отдельным контролем качества'
    split = m.split_records(records())
    features = {}
    for groups in split.values():
        for g in groups:
            sign = 1 if g['label'] == 'violation' else -1
            features[g['sha256']] = {'times': np.array([0., 2., 4., 6., 8.]),
                                    'features': np.array([[sign * 3, i / 5] for i in range(5)])}
    first = m.train_experiment(split, features, tmp_path / 'first')
    changed = {k: dict(v) for k, v in features.items()}
    for g in split['test']:
        changed[g['sha256']] = {'times': features[g['sha256']]['times'],
                               'features': features[g['sha256']]['features'] * -1000}
    second = m.train_experiment(split, changed, tmp_path / 'second')
    assert first['selected_c'] == second['selected_c']
    assert first['threshold'] == second['threshold']
    a = m.WindowClassifier(tmp_path / 'first' / 'candidate.npz')
    b = m.WindowClassifier(tmp_path / 'second' / 'candidate.npz')
    assert a.coef == pytest.approx(b.coef, abs=1e-12)
    assert first['test']['videos'] == 8
    assert first['auto_promoted'] is False


def test_frozen_protocol_rejects_changed_data_in_the_same_experiment(tmp_path):
    assert importlib.util.find_spec('app.train_run'), 'Нужен запуск обучения'
    from app.train_run import freeze_protocol
    protocol = {'seed': 17, 'groups': ['a', 'b']}
    freeze_protocol(tmp_path, protocol)
    freeze_protocol(tmp_path, protocol)
    with pytest.raises(ValueError, match='измен'):
        freeze_protocol(tmp_path, {'seed': 17, 'groups': ['a', 'b', 'c']})

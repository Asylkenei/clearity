import base64
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app


def test_jury_api_returns_top_level_contract_and_deletes_upload(tmp_path):
    # Slow neural inference is replaced at its boundary; real upload/store/API run.
    class Detector:
        def analyze(self, path, deadline):
            assert path.is_file()
            return {'verdict': 'no_violation', 'violations': [], 'review_required': True}

    with TestClient(create_app(api_key='jury-test', data_dir=tmp_path,
                               public_mode=True, detector=Detector())) as client:
        assert client.get('/health').json()['analysis_ready'] is True
        assert client.post('/api/v1/analyze', files={'file': ('a.mov', b'video')}).status_code == 401
        response = client.post('/api/v1/analyze', headers={'X-API-Key': 'jury-test'},
                               files={'file': ('a.mov', b'video')})
        assert response.status_code == 202
        for _ in range(100):
            job = client.get('/api/v1/jobs/' + response.json()['job_id'],
                             headers={'X-API-Key': 'jury-test'}).json()
            if job['status'] != 'processing':
                break
            time.sleep(.02)
        assert job['status'] == 'done'
        assert job['verdict'] == 'no_violation'
        assert job['violations'] == []
        assert list((tmp_path / 'public' / 'uploads').glob('*.mov')) == []


def test_trained_detector_extracts_real_jpeg_and_groups_adjacent_signals(tmp_path):
    from app.detector import TrainedDetector
    from app.features import FrameEncoder
    from app.training import WindowClassifier
    path = tmp_path / 'video.mov'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
    for i in range(60):
        frame = np.zeros((120, 160, 3), np.uint8)
        cv2.rectangle(frame, (i, 30), (i + 20, 60), (20, 200, 80), -1)
        writer.write(frame)
    writer.release()
    weights = tmp_path / 'weights.npz'
    np.savez(weights, mean=np.zeros(3072), scale=np.ones(3072), coef=np.zeros(3072),
             intercept=5., threshold=.55)
    detector = TrainedDetector(FrameEncoder(), WindowClassifier(weights))
    result = detector.analyze(path, time.monotonic() + 20)
    assert result['verdict'] == 'violation'
    assert len(result['violations']) == 1
    event = result['violations'][0]
    assert 0 <= event['timecode_sec'] < 6
    assert 1 <= len(event['evidence_frames']) <= 3
    assert base64.b64decode(event['evidence_frames'][0]['image_base64']).startswith(b'\xff\xd8')
    assert event['normative_source']['url'].startswith('https://adilet.zan.kz/')
    assert event['normative_source']['relationship'] == 'context_only'
    assert result['classification_validated'] is False


def test_detector_failure_never_becomes_no_violation(tmp_path):
    class BrokenDetector:
        def analyze(self, path, deadline):
            raise ValueError('Unreadable video')
    with TestClient(create_app(api_key='jury-test', data_dir=tmp_path,
                               public_mode=True, detector=BrokenDetector())) as client:
        response = client.post('/api/v1/analyze', headers={'X-API-Key': 'jury-test'},
                               files={'file': ('broken.mov', b'not video')})
        for _ in range(100):
            job = client.get('/api/v1/jobs/' + response.json()['job_id'],
                             headers={'X-API-Key': 'jury-test'}).json()
            if job['status'] != 'processing':
                break
            time.sleep(.02)
        assert job['status'] == 'failed'
        assert 'verdict' not in job

"""Video timestamps, local-only transport and failure semantics, without big weights."""
import base64
import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

HEADERS = {'X-API-Key': 'inspection-test-key'}


def module():
    assert importlib.util.find_spec('app.inspection'), 'Нужен локальный анализ видео'
    from app import inspection
    return inspection


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / 'sample.mov'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
    assert writer.isOpened()
    for i in range(220):
        writer.write(np.full((120, 160, 3), i, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def engine():
    # Replace only slow neural inference. Real HTTP transport and parsing stay in use.
    state = {'content': json.dumps({'events': [{
        'category': 'cash_handling', 'status': 'possible_concern',
        'observation': 'Учебное наблюдение', 'reason': 'Проверить вручную',
        'frame_indices': [1, 2]}]}), 'finish_reason': 'stop', 'http_status': 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            for item in payload['messages'][-1]['content']:
                if item['type'] == 'image_url':
                    url = item['image_url']['url']
                    assert url.startswith('data:image/jpeg;base64,')
                    image = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(',')[1]), np.uint8), 1)
                    assert image is not None
            data = json.dumps({'choices': [{'finish_reason': state['finish_reason'],
                               'message': {'content': state['content']}}]}).encode()
            self.send_response(state['http_status']); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state['port'] = server.server_port
    yield state
    server.shutdown(); server.server_close(); thread.join(2)


def test_sampling_covers_beginning_middle_and_end_with_decoder_times(movie):
    windows, meta = module().sample_windows(movie)
    times = [f['timestamp_sec'] for window in windows for f in window]
    assert len(windows) == 2
    assert times == pytest.approx([0, 10, 18, 20, 21, 21.8], abs=0.05)
    assert meta['duration_sec'] == 22
    assert meta['sampled_frames'] == 6


def test_local_model_returns_review_candidates_with_actual_evidence(movie, engine):
    m = module()
    result = m.inspect_video(movie, m.LocalModelClient(port=engine['port']))
    assert result['model_analysis_performed'] is True
    assert result['review_required'] is True
    assert result['violation_detection_validated'] is False
    assert 'verdict' not in result
    assert len(result['windows']) == 2
    event = result['windows'][0]['events'][0]
    assert event['timecode_sec'] == 10
    assert [f['timestamp_sec'] for f in event['evidence_frames']] == [10, 18]
    assert event['status'] == 'possible_concern'
    assert 'normative_source' not in event


@pytest.mark.parametrize('content,finish', [
    ('not json', 'stop'),
    ('{"events":[]}', 'length'),
    (json.dumps({'events': [{'category': 'invented', 'status': 'possible_concern',
       'observation': 'Something', 'reason': 'Something', 'frame_indices': [0]}]}), 'stop'),
    (json.dumps({'events': [{'category': 'other', 'status': 'observation',
       'observation': 'Something', 'reason': '', 'frame_indices': [99]}]}), 'stop'),
])
def test_invalid_or_truncated_model_output_is_an_error(movie, engine, content, finish):
    engine.update(content=content, finish_reason=finish)
    m = module()
    with pytest.raises(m.ModelError):
        m.inspect_video(movie, m.LocalModelClient(port=engine['port']))


def test_empty_events_never_means_no_violation(movie, engine):
    engine['content'] = '{"events":[]}'
    m = module()
    result = m.inspect_video(movie, m.LocalModelClient(port=engine['port']))
    assert result['review_required'] is True
    assert 'verdict' not in result
    assert result['windows'][0]['events'] == []


@pytest.mark.parametrize('category', ['other', 'unclear'])
def test_unrelated_or_unclear_objects_cannot_be_flagged_as_cash_workplace_concerns(movie, engine, category):
    engine['content'] = json.dumps({'events': [{
        'category': category, 'status': 'possible_concern',
        'observation': 'Красный квадрат на белом фоне', 'reason': 'Непонятно, что это',
        'frame_indices': [0]}]})
    m = module()
    result = m.inspect_video(movie, m.LocalModelClient(port=engine['port']))
    assert result['windows'][0]['events'][0]['status'] == 'unclear'
    assert result['review_required'] is True


def test_offline_model_failure_becomes_failed_job_and_upload_is_removed(movie, engine, tmp_path):
    engine['http_status'] = 503
    m = module()
    runtime = tmp_path / 'runtime'
    with TestClient(create_app(api_key=HEADERS['X-API-Key'], data_dir=runtime, public_mode=True,
                              model_client=m.LocalModelClient(port=engine['port']))) as client:
        with movie.open('rb') as stream:
            submitted = client.post('/api/v1/inspect', headers=HEADERS,
                                    files={'file': ('sample.mov', stream, 'video/quicktime')})
        assert submitted.status_code == 202
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get('/api/v1/jobs/' + submitted.json()['job_id'], headers=HEADERS).json()
            if job['status'] != 'processing': break
            time.sleep(0.02)
        assert job['status'] == 'failed'
        assert job['error']['code'] == 'LOCAL_MODEL_FAILED'
        assert 'verdict' not in job
    assert not list((runtime / 'public' / 'uploads').glob('*.mov'))


def test_expired_budget_stops_without_a_clean_verdict(movie, engine):
    m = module()
    with pytest.raises(m.ModelError, match='время'):
        m.inspect_video(movie, m.LocalModelClient(port=engine['port']), deadline=time.monotonic() - 1)

"""Проверяем реальную загрузку и чтение видео, без облака и подмен модели."""

import base64
import importlib
import importlib.util
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


KEY = "test-key-used-only-in-tests"
HEADERS = {"X-API-Key": KEY}


def test_application_can_start(tmp_path):
    with TestClient(factory()(api_key=KEY, data_dir=tmp_path / "runtime")) as c:
        assert c.get("/health").status_code == 200


def factory():
    # Первый запуск должен показать, что сервер ещё не написан.
    assert importlib.util.find_spec("app.main") is not None, "Нужно написать app/main.py"
    return importlib.import_module("app.main").create_app


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / "sample.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened(), "Не удалось создать учебное видео"
    for i in range(30):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i * 3, 25), (i * 3 + 25, 60), (50, 180, 240), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def client(tmp_path):
    with TestClient(factory()(api_key=KEY, data_dir=tmp_path / "runtime")) as c:
        yield c


def wait_for_job(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS)
        assert response.status_code == 200
        job = response.json()
        if job["status"] != "processing":
            return job
        time.sleep(0.02)
    pytest.fail("Задание не завершилось за 10 секунд")


def test_health_explains_that_analysis_is_not_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["analysis_ready"] is False


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_upload_requires_correct_key(client, headers):
    response = client.post("/api/v1/preview", headers=headers,
                           files={"file": ("sample.mov", b"abc", "video/quicktime")})
    assert response.status_code == 401


def test_job_result_requires_key(client):
    assert client.get("/api/v1/jobs/unknown").status_code == 401


def test_unknown_job_returns_404(client):
    assert client.get("/api/v1/jobs/unknown", headers=HEADERS).status_code == 404


def test_non_mov_is_rejected(client):
    response = client.post("/api/v1/preview", headers=HEADERS,
                           files={"file": ("note.txt", b"abc", "text/plain")})
    assert response.status_code == 415


def test_empty_file_is_rejected(client):
    response = client.post("/api/v1/preview", headers=HEADERS,
                           files={"file": ("empty.mov", b"", "video/quicktime")})
    assert response.status_code == 400


def test_large_upload_is_rejected_and_partial_file_removed(tmp_path):
    data = tmp_path / "runtime"
    with TestClient(factory()(api_key=KEY, data_dir=data, max_upload_bytes=8)) as c:
        response = c.post("/api/v1/preview", headers=HEADERS,
                          files={"file": ("large.mov", b"123456789", "video/quicktime")})
        assert response.status_code == 413
        assert list((data / "uploads").glob("*.mov")) == []


def test_real_movie_produces_metadata_and_jpeg_frames_without_verdict(client, movie):
    with movie.open("rb") as video:
        response = client.post("/api/v1/preview", headers=HEADERS,
                               files={"file": ("sample.MOV", video, "video/quicktime")})
    assert response.status_code == 202
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "done"
    assert job["kind"] == "preview"
    assert "verdict" not in job
    result = job["result"]
    assert result["analysis_performed"] is False
    assert result["width"] == 160
    assert result["height"] == 120
    assert result["duration_sec"] == pytest.approx(3, abs=0.1)
    assert len(result["frames"]) == 3
    timestamps = [f["timestamp_sec"] for f in result["frames"]]
    assert timestamps == sorted(timestamps)
    assert all(0 <= t < 3 for t in timestamps)
    for frame in result["frames"]:
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(frame["image_base64"]),
                                            dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[:2] == (120, 160)


def test_corrupt_video_is_failed_and_never_no_violation(client):
    response = client.post("/api/v1/preview", headers=HEADERS,
                           files={"file": ("broken.mov", b"this is not video", "video/quicktime")})
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]["code"] == "VIDEO_UNREADABLE"
    assert "verdict" not in job


def test_analysis_endpoint_refuses_to_invent_a_result(client, movie):
    with movie.open("rb") as video:
        response = client.post("/api/v1/analyze", headers=HEADERS,
                               files={"file": ("sample.mov", video, "video/quicktime")})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "VIOLATION_DETECTOR_NOT_VALIDATED"


def test_completed_job_survives_app_restart(tmp_path, movie):
    data = tmp_path / "runtime"
    with TestClient(factory()(api_key=KEY, data_dir=data)) as c:
        with movie.open("rb") as video:
            response = c.post("/api/v1/preview", headers=HEADERS,
                              files={"file": ("sample.mov", video, "video/quicktime")})
        job_id = response.json()["job_id"]
        assert wait_for_job(c, job_id)["status"] == "done"
    with TestClient(factory()(api_key=KEY, data_dir=data)) as c:
        assert c.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()["status"] == "done"

import time
import importlib.util
import json
import subprocess
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.store import JobStore
from dotenv import dotenv_values


PUBLIC_KEY = "test-public-key"
HEADERS = {"X-API-Key": PUBLIC_KEY}


@pytest.mark.skipif(shutil.which("powershell") is None, reason="Windows PowerShell launcher")
def test_tunnel_launcher_checks_and_refuses_a_local_review_server():
    checked = threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            checked.set()
            data = json.dumps({"status": "ok", "public_mode": False}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        script = Path(__file__).resolve().parent.parent / "start-tunnel.ps1"
        result = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                 str(script), "-Port", str(server.server_port)], capture_output=True, timeout=20)
        assert checked.is_set(), "Перед запуском туннеля нужно проверить режим сервера"
        assert result.returncode != 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_public_key_setup_preserves_local_key_and_reuses_public_key(tmp_path):
    assert importlib.util.find_spec("app.public_setup") is not None, "Нужна настройка отдельного публичного ключа"
    from app.public_setup import ensure_public_key
    env_file = tmp_path / ".env"
    env_file.write_text("FORTE_API_KEY=local-admin-secret\nOTHER_SETTING=keep-me\n", encoding="utf-8")
    key = ensure_public_key(env_file)
    assert len(key) >= 32
    assert key != "local-admin-secret"
    values = dotenv_values(env_file)
    assert values["FORTE_API_KEY"] == "local-admin-secret"
    assert values["OTHER_SETTING"] == "keep-me"
    assert values["FORTE_PUBLIC_API_KEY"] == key
    before = env_file.read_bytes()
    assert ensure_public_key(env_file) == key
    assert env_file.read_bytes() == before


def test_public_key_setup_replaces_a_reused_admin_key(tmp_path):
    assert importlib.util.find_spec("app.public_setup") is not None, "Нужна настройка отдельного публичного ключа"
    from app.public_setup import ensure_public_key
    env_file = tmp_path / ".env"
    env_file.write_text("FORTE_API_KEY=local-secret\nFORTE_PUBLIC_API_KEY=local-secret\n", encoding="utf-8")
    key = ensure_public_key(env_file)
    assert key != "local-secret"
    assert dotenv_values(env_file)["FORTE_API_KEY"] == "local-secret"


def test_public_app_is_available_without_reading_the_training_manifest(tmp_path):
    dataset = tmp_path / "private-dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("intentionally invalid JSON")
    with TestClient(create_app(api_key=PUBLIC_KEY, data_dir=tmp_path / "runtime",
                               dataset_dir=dataset, public_mode=True)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["public_mode"] is True
    assert not list((tmp_path / "runtime").rglob("annotations.sqlite3"))


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(api_key=PUBLIC_KEY, data_dir=tmp_path / "runtime", public_mode=True)) as c:
        yield c


@pytest.mark.parametrize("method,path", [
    ("GET", "/review"), ("GET", "/api/v1/review/dataset"),
    ("GET", "/api/v1/review/videos/X-1"), ("GET", "/api/v1/review/categories"),
    ("POST", "/api/v1/review/categories"), ("PUT", "/api/v1/review/annotations/X-1"),
    ("GET", "/api/v1/review/export"), ("GET", "/static/review.html"),
    ("GET", "/static/review.js"), ("GET", "/docs"), ("GET", "/openapi.json"),
])
def test_public_link_does_not_expose_review_or_documentation(client, method, path):
    assert client.request(method, path, headers=HEADERS, json={}).status_code == 404


def test_public_page_offers_upload_instead_of_the_private_review_link(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/review"' not in response.text
    assert 'id="upload-form"' in response.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "local-admin-key"}])
def test_public_upload_requires_the_public_key(client, headers):
    response = client.post("/api/v1/preview", headers=headers,
                           files={"file": ("sample.mov", b"abc", "video/quicktime")})
    assert response.status_code == 401


def test_public_jobs_are_separate_from_existing_local_jobs(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    JobStore(runtime / "jobs.sqlite3").put({"job_id": "private-job", "status": "done"})
    with TestClient(create_app(api_key=PUBLIC_KEY, data_dir=runtime, public_mode=True)) as c:
        assert c.get("/api/v1/jobs/private-job", headers=HEADERS).status_code == 404


def test_public_preview_reads_new_video_but_does_not_invent_an_ai_result(client, tmp_path):
    path = tmp_path / "public-sample.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(30):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    with path.open("rb") as source:
        response = client.post("/api/v1/preview", headers=HEADERS,
                               files={"file": (path.name, source, "video/quicktime")})
    assert response.status_code == 202
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get('/api/v1/jobs/' + response.json()["job_id"], headers=HEADERS).json()
        if job["status"] != "processing":
            break
        time.sleep(0.02)
    assert job["status"] == "done"
    assert job["result"]["analysis_performed"] is False
    assert len(job["result"]["frames"]) == 3
    assert client.post("/api/v1/analyze", headers=HEADERS).status_code == 503

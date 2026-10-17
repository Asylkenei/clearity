import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


KEY = "review-test-key"
HEADERS = {"X-API-Key": KEY}


def test_review_page_is_available(tmp_path):
    with TestClient(create_app(api_key=KEY, data_dir=tmp_path / "runtime")) as client:
        assert client.get("/review").status_code == 200


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    records = []
    for number, label, mark in [(1, "violation", 1.0), (2, "no_violation", None)]:
        video = folder / f"X-{number}.mov"
        video.write_bytes(b"local video bytes for range test")
        records.append({"video_id": f"X-{number}", "filename": video.name, "path": str(video),
                        "label": label, "label_timecode_sec": mark, "duration_sec_estimate": 3.0,
                        "sha256": "same-source", "width": 160, "height": 120, "fps": 10})
    (folder / "manifest.json").write_text(json.dumps({"records": records, "summary": {"videos": 2},
        "warnings": [], "duplicate_groups": []}), encoding="utf-8")
    return folder


@pytest.fixture
def client(dataset, tmp_path):
    with TestClient(create_app(api_key=KEY, data_dir=tmp_path / "runtime", dataset_dir=dataset)) as c:
        yield c


@pytest.mark.parametrize("method,url", [
    ("GET", "/api/v1/review/dataset"), ("GET", "/api/v1/review/categories"),
    ("GET", "/api/v1/review/videos/X-1"), ("GET", "/api/v1/review/export"),
    ("PUT", "/api/v1/review/annotations/X-1"), ("POST", "/api/v1/review/categories"),
])
def test_review_data_requires_key(client, method, url):
    assert client.request(method, url, json={}).status_code == 401


def test_dataset_has_original_labels_but_no_filesystem_paths(client):
    payload = client.get("/api/v1/review/dataset", headers=HEADERS).json()
    assert len(payload["records"]) == 2
    assert payload["records"][0]["label"] == "violation"
    assert payload["records"][0]["label_timecode_sec"] == 1.0
    assert payload["records"][0]["annotation"]["review_state"] == "not_started"
    assert "path" not in payload["records"][0]
    assert payload["records"][0]["same_content_ids"] == ["X-2"]
    assert payload["progress"]["reviewed"] == 0


def test_video_supports_partial_requests_and_unknown_ids_are_404(client):
    response = client.get("/api/v1/review/videos/X-1", headers={**HEADERS, "Range": "bytes=0-4"})
    assert response.status_code == 206
    assert response.content == b"local"
    assert client.get("/api/v1/review/videos/X-999", headers=HEADERS).status_code == 404


def test_video_cannot_escape_the_dataset_directory(client, dataset, tmp_path):
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    outside = tmp_path / "outside.mov"
    outside.write_bytes(b"private")
    manifest["records"][0]["path"] = str(outside)
    manifest_path.write_text(json.dumps(manifest))
    with TestClient(create_app(api_key=KEY, data_dir=tmp_path / "other-runtime", dataset_dir=dataset)) as c:
        response = c.get("/api/v1/review/videos/X-1", headers=HEADERS)
    assert response.status_code == 404
    assert b"private" not in response.content


def test_category_is_a_draft_and_duplicate_name_is_rejected(client):
    response = client.post("/api/v1/review/categories", headers=HEADERS,
                           json={"name": "Передача предмета", "definition": "Предмет переходит между людьми"})
    assert response.status_code == 201
    assert response.json()["normative_status"] == "not_checked"
    duplicate = client.post("/api/v1/review/categories", headers=HEADERS,
                            json={"name": "  передача предмета  "})
    assert duplicate.status_code == 409


def test_annotation_survives_restart_and_does_not_modify_source_labels(client, dataset, tmp_path):
    before = (dataset / "manifest.json").read_bytes()
    category = client.post("/api/v1/review/categories", headers=HEADERS,
                           json={"name": "Учебное наблюдение"}).json()
    annotation = {"observation": "Человек перемещает предмет по столу.", "category_id": category["id"],
                  "visibility": "clear", "review_state": "reviewed", "event_start_sec": 0.5,
                  "event_end_sec": 1.5, "notes": "Нормативное основание ещё не проверено."}
    response = client.put("/api/v1/review/annotations/X-1", headers=HEADERS, json=annotation)
    assert response.status_code == 200
    assert (dataset / "manifest.json").read_bytes() == before
    with TestClient(create_app(api_key=KEY, data_dir=tmp_path / "runtime", dataset_dir=dataset)) as c:
        data = c.get("/api/v1/review/dataset", headers=HEADERS).json()
    assert data["records"][0]["annotation"]["observation"] == annotation["observation"]
    assert data["progress"]["reviewed"] == 1
    assert data["records"][1]["annotation"]["review_state"] == "not_started"


@pytest.mark.parametrize("body", [
    {"observation": "", "review_state": "reviewed"},
    {"observation": "Действие видно", "review_state": "reviewed", "category_id": "unknown"},
    {"observation": "Действие видно", "review_state": "reviewed"},
    {"event_start_sec": 2, "event_end_sec": 1},
    {"event_start_sec": -1}, {"event_end_sec": 99},
    {"label": "no_violation"},
])
def test_incomplete_or_invalid_review_is_rejected(client, body):
    assert client.put("/api/v1/review/annotations/X-1", headers=HEADERS, json=body).status_code == 422


def test_unclear_can_be_saved_without_inventing_a_category(client):
    body = {"observation": "Предмет закрыт рукой.", "visibility": "unclear", "review_state": "needs_review"}
    assert client.put("/api/v1/review/annotations/X-1", headers=HEADERS, json=body).status_code == 200


def test_normal_clip_can_be_reviewed_without_violation_category(client):
    body = {"observation": "Видна обычная последовательность действий.", "review_state": "reviewed"}
    assert client.put("/api/v1/review/annotations/X-2", headers=HEADERS, json=body).status_code == 200


def test_export_keeps_empty_reviews_and_unchecked_categories(client):
    export = client.get("/api/v1/review/export", headers=HEADERS)
    assert export.status_code == 200
    data = export.json()
    assert len(data["records"]) == 2
    assert len(data["categories"]) >= 1
    assert all(c["normative_status"] == "not_checked" for c in data["categories"])
    assert "path" not in data["records"][0]

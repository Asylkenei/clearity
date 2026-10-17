"""SQLite хранит статусы и результаты заданий между перезапусками."""

import json
import sqlite3
from pathlib import Path


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            # Задание, прерванное выключением сервера, нельзя объявлять успешным.
            rows = db.execute("SELECT id, payload FROM jobs").fetchall()
            for job_id, payload in rows:
                job = json.loads(payload)
                if job["status"] == "processing":
                    job["status"] = "failed"
                    job["error"] = {"code": "SERVER_RESTARTED", "message": "Сервер перезапустился. Загрузите видео повторно."}
                    db.execute("UPDATE jobs SET payload = ? WHERE id = ?",
                               (json.dumps(job, ensure_ascii=False), job_id))

    def put(self, job: dict):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO jobs (id, payload) VALUES (?, ?)",
                       (job["job_id"], json.dumps(job, ensure_ascii=False)))

    def get(self, job_id: str) -> dict | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

"""Ручной локальный разбор: исходные метки не меняются, наблюдения хранятся отдельно."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CategoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    definition: str = Field(default="", max_length=2000)


class AnnotationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    observation: str = Field(default="", max_length=3000)
    category_id: str | None = None
    visibility: Literal["clear", "partial", "unclear"] = "clear"
    review_state: Literal["draft", "reviewed", "needs_review"] = "draft"
    event_start_sec: float | None = Field(default=None, ge=0)
    event_end_sec: float | None = Field(default=None, ge=0)
    notes: str = Field(default="", max_length=3000)

    @model_validator(mode="after")
    def check_review(self):
        self.observation = self.observation.strip()
        if self.review_state == "reviewed" and not self.observation:
            raise ValueError("Для завершения просмотра запишите видимое действие.")
        if self.event_start_sec is not None and self.event_end_sec is not None:
            if self.event_start_sec > self.event_end_sec:
                raise ValueError("Начало события должно быть раньше конца.")
        return self


# Это заготовки групп наблюдений, а не обнаруженные типы нарушения.
STARTER_GROUPS = [
    ("cash_transfer", "Передача и перемещение наличных", "Кто, что и кому передаёт; откуда и куда перемещаются банкноты или монеты."),
    ("cash_check", "Пересчёт и проверка наличных", "Наблюдаемая последовательность пересчёта и проверки; какие действия различимы."),
    ("valuables_storage", "Хранение и наблюдение за ценностями", "Где находятся ценности и кто остаётся рядом с ними."),
    ("workspace_access", "Доступ к рабочей зоне", "Кто входит в рабочую зону или получает доступ к предметам в ней."),
    ("collection", "Действия при инкассации", "Наблюдаемые передачи сумок, упаковок и ценностей при участии инкассаторов."),
]


class ReviewStore:
    def __init__(self, dataset_dir: Path, db_path: Path):
        self.dataset_dir = dataset_dir.resolve()
        self.db_path = db_path
        manifest_path = self.dataset_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
            "records": [], "warnings": ["Каталог не найден. Сначала выполните python -m app.dataset ПАПКА."],
        }
        self.records = {r["video_id"]: r for r in self.manifest["records"]}
        with sqlite3.connect(db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS categories (id TEXT PRIMARY KEY, name_key TEXT UNIQUE, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS annotations (video_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            for category_id, name, definition in STARTER_GROUPS:
                category = {"id": category_id, "name": name, "definition": definition,
                            "origin": "starter_observation_group", "normative_status": "not_checked"}
                db.execute("INSERT OR IGNORE INTO categories VALUES (?, ?, ?)",
                           (category_id, name.casefold(), json.dumps(category, ensure_ascii=False)))

    def record(self, video_id: str) -> dict:
        record = self.records.get(video_id)
        if record is None:
            raise HTTPException(404, "Видео не найдено в каталоге.")
        return record

    def video_path(self, video_id: str) -> Path:
        path = Path(self.record(video_id)["path"]).resolve()
        if not path.is_relative_to(self.dataset_dir) or path.suffix.lower() != ".mov" or not path.is_file():
            raise HTTPException(404, "Видео недоступно в папке датасета.")
        return path

    def categories(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("SELECT payload FROM categories ORDER BY rowid").fetchall()
        return [json.loads(row[0]) for row in rows]

    def add_category(self, data: CategoryInput) -> dict:
        name = " ".join(data.name.split())
        if not name:
            raise HTTPException(422, "Укажите название категории.")
        category = {"id": uuid.uuid4().hex, "name": name, "definition": data.definition.strip(),
                    "origin": "human_draft", "normative_status": "not_checked"}
        try:
            with sqlite3.connect(self.db_path) as db:
                db.execute("INSERT INTO categories VALUES (?, ?, ?)",
                           (category["id"], name.casefold(), json.dumps(category, ensure_ascii=False)))
        except sqlite3.IntegrityError:
            raise HTTPException(409, "Категория с таким названием уже есть.") from None
        return category

    def save_annotation(self, video_id: str, data: AnnotationInput) -> dict:
        record = self.record(video_id)
        ids = {c["id"] for c in self.categories()}
        if data.category_id and data.category_id not in ids:
            raise HTTPException(422, "Выберите существующую категорию.")
        if data.review_state == "reviewed" and record["label"] == "violation" and not data.category_id:
            raise HTTPException(422, "Выберите рабочую категорию или сохраните статус «Нужно уточнить».")
        duration = record.get("duration_sec_estimate")
        for value in (data.event_start_sec, data.event_end_sec):
            if value is not None and duration is not None and value > duration:
                raise HTTPException(422, "Время события выходит за длительность видео.")
        annotation = {**data.model_dump(), "updated_at": datetime.now(timezone.utc).isoformat()}
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT OR REPLACE INTO annotations VALUES (?, ?)",
                       (video_id, json.dumps(annotation, ensure_ascii=False)))
        return annotation

    def dataset(self) -> dict:
        with sqlite3.connect(self.db_path) as db:
            annotations = {row[0]: json.loads(row[1]) for row in db.execute("SELECT video_id, payload FROM annotations")}
        records = []
        for record in self.records.values():
            public = {key: value for key, value in record.items() if key not in ("path", "sha256")}
            public["same_content_ids"] = [r["video_id"] for r in self.records.values()
                                          if r["video_id"] != record["video_id"] and r.get("sha256") and r.get("sha256") == record.get("sha256")]
            public["annotation"] = annotations.get(record["video_id"], {
                **AnnotationInput().model_dump(), "review_state": "not_started",
            })
            records.append(public)
        return {
            "records": records, "warnings": self.manifest.get("warnings", []),
            "summary": {"videos": len(records), "violation": sum(r["label"] == "violation" for r in records),
                        "no_violation": sum(r["label"] == "no_violation" for r in records)},
            "progress": {state: sum(r["annotation"]["review_state"] == state for r in records)
                         for state in ("not_started", "draft", "reviewed", "needs_review")},
        }

"""Учебный этап 1: сервер, загрузка MOV и подготовка кадров."""

import logging
import os
import secrets
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import dotenv_values
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from app.store import JobStore
from app.video import prepare_video
from app.review import AnnotationInput, CategoryInput, ReviewStore


ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger(__name__)


def create_app(api_key: str | None = None, data_dir: Path | None = None,
               max_upload_bytes: int = 100 * 1024 * 1024,
               dataset_dir: Path | None = None) -> FastAPI:
    """Создаём приложение. Параметры позволяют проверять его в изоляции."""
    config = dotenv_values(ROOT / ".env")
    api_key = api_key or os.environ.get("FORTE_API_KEY") or config.get("FORTE_API_KEY")
    if not api_key:
        raise RuntimeError("Не задан FORTE_API_KEY. Сначала выполните setup.ps1.")
    data_dir = Path(data_dir or os.environ.get("FORTE_DATA_DIR") or ROOT / "runtime")
    uploads = data_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    store = JobStore(data_dir / "jobs.sqlite3")
    review = ReviewStore(Path(dataset_dir or os.environ.get("FORTE_DATASET_DIR") or ROOT.parent.parent / "work" / "dataset"),
                        data_dir / "annotations.sqlite3")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-preview")

    @asynccontextmanager
    async def lifespan(_app):
        yield
        executor.shutdown(wait=True)

    app = FastAPI(title="Forte Vision — учебный проект", version="0.2.0", lifespan=lifespan)
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(key: Annotated[str | None, Depends(key_header)]):
        if not key or not secrets.compare_digest(key, api_key):
            raise HTTPException(401, "Неверный или отсутствующий ключ доступа.")

    def process_preview(job: dict, path: Path):
        try:
            job["result"] = prepare_video(path)
            job["status"] = "done"
        except (ValueError, OSError) as exc:
            job["status"] = "failed"
            job["error"] = {"code": "VIDEO_UNREADABLE", "message": str(exc)}
        except Exception:
            log.exception("Preview failed for job %s", job["job_id"])
            job["status"] = "failed"
            job["error"] = {"code": "PROCESSING_FAILED", "message": "Не удалось подготовить видео."}
        finally:
            store.put(job)
            # После подготовки сохраняем только обзорные кадры в локальной БД.
            path.unlink(missing_ok=True)

    @app.get("/health")
    def health():
        return {"status": "ok", "stage": "video_preparation", "analysis_ready": False}

    @app.post("/api/v1/preview", status_code=202, dependencies=[Depends(require_key)])
    async def preview(file: Annotated[UploadFile, File()]):
        """Дополнительный учебный API. Он не выдаёт вердикт о нарушениях."""
        if Path(file.filename or "").suffix.lower() != ".mov":
            await file.close()
            raise HTTPException(415, "Выберите видео в формате .mov.")

        job_id = uuid.uuid4().hex
        path = uploads / f"{job_id}.mov"
        size = 0
        try:
            with path.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(413, "Файл превышает допустимый размер.")
                    target.write(chunk)
            if size == 0:
                raise HTTPException(400, "Выбранный файл пуст.")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        job = {"job_id": job_id, "kind": "preview", "status": "processing",
               "filename": Path(file.filename or "video.mov").name, "size_bytes": size}
        store.put(job)
        executor.submit(process_preview, dict(job), path)
        return {"job_id": job_id, "status": "processing", "kind": "preview"}

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def get_job(job_id: str):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Задание не найдено.")
        return job

    @app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
    def analyze():
        # Нельзя возвращать no_violation, если анализ ещё не реализован.
        raise HTTPException(503, detail={
            "code": "MODEL_NOT_CONFIGURED",
            "message": "Проверка нарушений появится на следующем этапе. Сейчас доступна подготовка видео.",
        })

    @app.get("/api/v1/review/dataset", dependencies=[Depends(require_key)])
    def review_dataset():
        return review.dataset()

    @app.get("/api/v1/review/categories", dependencies=[Depends(require_key)])
    def review_categories():
        return review.categories()

    @app.post("/api/v1/review/categories", status_code=201, dependencies=[Depends(require_key)])
    def add_review_category(data: CategoryInput):
        return review.add_category(data)

    @app.put("/api/v1/review/annotations/{video_id}", dependencies=[Depends(require_key)])
    def save_review_annotation(video_id: str, data: AnnotationInput):
        return review.save_annotation(video_id, data)

    @app.get("/api/v1/review/videos/{video_id}", dependencies=[Depends(require_key)])
    def review_video(video_id: str):
        return FileResponse(review.video_path(video_id), media_type="video/mp4")

    @app.get("/api/v1/review/export", dependencies=[Depends(require_key)])
    def export_review():
        return {"format_version": 1, **review.dataset(), "categories": review.categories(),
                "note": "Ручные наблюдения. Рабочие категории ещё не подтверждены нормативными источниками."}

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    @app.get("/review", include_in_schema=False)
    def review_page():
        return FileResponse(ROOT / "static" / "review.html")

    return app

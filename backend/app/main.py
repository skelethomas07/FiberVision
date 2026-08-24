from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import analyses, images, reviews
from .config import get_settings
from .db import Base, SessionLocal, engine
from .queue import RQAnalysisQueue
from .storage import build_storage


def create_app(*, session_factory=None, storage=None, queue=None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="SEM Fiber Analysis API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.Session = session_factory or SessionLocal
    app.state.storage = storage or build_storage(settings)
    app.state.queue = queue or RQAnalysisQueue(settings.redis_url)
    app.state.model_version = settings.model_version
    app.include_router(images.router)
    app.include_router(analyses.router)
    app.include_router(reviews.router)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app


app = create_app()

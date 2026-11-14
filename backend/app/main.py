import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import make_url
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.database.session import (
    create_database,
    create_offline_database,
    initialize_database,
    initialize_offline_database,
)
from app.logging_config import configure_logging
from app.routes.api import router
from app.schemas import ErrorResponse
from app.services.inference import InferenceProvider, ModelState, UnavailableInferenceProvider
from app.services.inference_contracts import FaceDetectionProvider
from app.services.member1_face_provider import Member1FaceDetectionProvider

logger = logging.getLogger(__name__)


def _offline_database_url(settings: Settings) -> str:
    if settings.offline_database_url:
        return settings.offline_database_url
    primary_url = make_url(settings.database_url)
    if (
        primary_url.drivername.startswith("sqlite")
        and primary_url.database
        and primary_url.database != ":memory:"
    ):
        from pathlib import Path

        primary_path = Path(primary_url.database)
        offline_path = primary_path.with_name(
            f"{primary_path.stem}_offline{primary_path.suffix or '.db'}"
        )
        return f"sqlite:///{offline_path.as_posix()}"
    return "sqlite:///./hid_offline.db"


def create_app(
    settings: Settings | None = None,
    inference_provider: InferenceProvider | None = None,
    clock: Callable[[], datetime] | None = None,
    face_detection_provider: FaceDetectionProvider | None = None,
) -> FastAPI:
    active_settings = settings or get_settings()
    engine, session_factory = create_database(active_settings.database_url)
    offline_engine, offline_session_factory = create_offline_database(
        _offline_database_url(active_settings)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_face_provider = face_detection_provider
        try:
            initialize_database(engine)
            initialize_offline_database(offline_engine)
            if active_face_provider is None and active_settings.face_detection_enabled:
                active_face_provider = Member1FaceDetectionProvider(
                    detector_model_path=active_settings.face_detection_model_path,
                    landmarker_model_path=active_settings.face_landmarker_model_path,
                )
            app.state.face_detection_provider = active_face_provider
            yield
        finally:
            try:
                if active_face_provider is not None:
                    close = getattr(active_face_provider, "close", None)
                    if callable(close):
                        close()
            finally:
                engine.dispose()
                offline_engine.dispose()

    app = FastAPI(title=active_settings.app_name, version="1.0.0", lifespan=lifespan)
    app.state.settings = active_settings
    app.state.session_factory = session_factory
    app.state.offline_engine = offline_engine
    app.state.offline_session_factory = offline_session_factory
    app.state.inference_provider = inference_provider or UnavailableInferenceProvider()
    app.state.face_detection_provider = face_detection_provider
    app.state.run_inference = run_in_threadpool
    app.state.clock = clock or (lambda: datetime.now(timezone.utc))

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": detail, "code": f"http_{exc.status_code}"},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, _exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed.", "code": "invalid_request"},
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(_request: Request, exc: SQLAlchemyError):
        logger.error("Database operation failed (%s)", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "Database operation failed.", "code": "database_error"},
        )

    @app.get("/health", responses={503: {"model": ErrorResponse}})
    def health():
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            logger.error("Health check database query failed (%s)", type(exc).__name__)
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "database": "ERROR"},
            )
        state = (
            ModelState.UNAVAILABLE.value
            if isinstance(app.state.inference_provider, UnavailableInferenceProvider)
            else "CONFIGURED_UNVERIFIED"
        )
        return {"status": "ok", "database": "ok", "inference": state}

    app.include_router(router)
    return app


configure_logging()
app = create_app()

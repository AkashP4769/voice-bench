"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import routes_health, routes_runs, routes_ui
from app.errors import AudioError, ProviderError, ProviderUnavailable
from app.ui.render import STATIC_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s: %(message)s",
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Voice Bench",
        version=__version__,
        description=(
            "Local benchmark harness for Phonon-2 and Whistle. "
            "Providers are reachable in-process or over an OpenAI-compatible "
            "HTTP endpoint, so a hosted provider is a configuration change."
        ),
    )

    # Audio the user cannot decode and providers that cannot run are user
    # errors, not server faults, so they get actionable status codes.
    @app.exception_handler(AudioError)
    async def _audio_error(request: Request, exc: AudioError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ProviderUnavailable)
    async def _provider_unavailable(
        request: Request, exc: ProviderUnavailable
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def _provider_error(request: Request, exc: ProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    app.include_router(routes_health.router)
    app.include_router(routes_runs.router)
    app.include_router(routes_ui.router)

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=False)


if __name__ == "__main__":
    main()

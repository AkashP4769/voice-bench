"""Health and provider discovery.

``/api/providers`` is the endpoint the UI builds its form from, and it is also
the place where the "never reveal a key" rule is enforced: it reports whether a
credential is configured as a boolean and names the environment variable, but
never reads or returns the value.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.deps import get_config, get_worker_pool
from app.config import Settings
from app.providers.registry import describe_configuration

router = APIRouter(tags=["providers"])


@router.get("/api/health")
def health(
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
) -> dict:
    return {
        "status": "ok",
        "providers_configured": len(settings.providers),
        "workers": pool.status(),
    }


@router.get("/api/providers")
def list_providers(
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
) -> dict:
    workers = {entry["provider"]: entry for entry in pool.status()}
    providers = []
    for name, config in settings.providers.items():
        entry = describe_configuration(config)
        entry["worker"] = workers.get(name)
        providers.append(entry)
    return {
        "providers": providers,
        "defaults": asdict(settings.defaults),
    }


@router.post("/api/providers/{provider}/warm")
def warm_provider(
    provider: str,
    settings: Settings = Depends(get_config),
    pool=Depends(get_worker_pool),
) -> dict:
    """Load a model now, so a later run is not slowed by first-use setup."""
    if provider not in settings.providers:
        known = ", ".join(sorted(settings.providers)) or "none"
        return {"provider": provider, "status": "error",
                "message": f"unknown provider (known: {known})"}
    outcomes = pool.warm([provider])
    status = outcomes.get(provider, "unknown")
    return {
        "provider": provider,
        "status": "ready" if status == "ready" else "error",
        "message": None if status == "ready" else status,
    }

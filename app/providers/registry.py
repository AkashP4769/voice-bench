"""Provider registry.

Two different jobs live here:

* ``capabilities_for`` describes a provider without starting its engine, so the
  API and the UI can render the right options cheaply.
* ``build_provider`` constructs an adapter, which happens inside a worker
  process where the engine is loaded.

Adapter modules are imported by name only when their provider type is used, so
a machine that has one engine installed never trips over the other's imports.
"""

from __future__ import annotations

import importlib
import logging

from app.config import ProviderConfig, Settings
from app.domain.provider import ProviderCapabilities
from app.errors import ProviderError

logger = logging.getLogger(__name__)

# Provider type -> "module:ClassName"
ADAPTERS: dict[str, str] = {
    "whistle": "app.providers.whistle:WhistleProvider",
    "phonon": "app.providers.phonon:PhononProvider",
}

HUMAN_NAMES: dict[str, str] = {
    "whistle": "Whistle (Cactus Compute)",
    "phonon": "Phonon-2 (Fermion Research)",
}


class UnknownProviderType(ProviderError):
    pass


def _load_class(provider_type: str):
    target = ADAPTERS.get(provider_type)
    if target is None:
        known = ", ".join(sorted(ADAPTERS))
        raise UnknownProviderType(
            f"unknown provider type {provider_type!r} (known types: {known})"
        )
    module_name, class_name = target.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def capabilities_for(config: ProviderConfig) -> ProviderCapabilities:
    """Declared capabilities, without importing or starting an engine."""
    return _load_class(config.type).capabilities


def build_provider(config: ProviderConfig, settings: Settings):
    """Instantiate an adapter. Loading the engine is a separate ``warmup()``."""
    return _load_class(config.type)(config, settings)


def describe_configuration(config: ProviderConfig) -> dict:
    """Provider metadata for the API, with no secret material in it."""
    capabilities = capabilities_for(config)
    return {
        "name": config.name,
        "type": config.type,
        "model": config.model,
        "display_name": HUMAN_NAMES.get(config.type, config.type),
        "transport": config.transport,
        "capabilities": capabilities.as_dict(),
        # A boolean only. The key value is never read, returned or rendered here.
        "credential_configured": config.credential_configured,
        "api_key_env": config.server.api_key_env if config.server else None,
        "base_url": config.server.base_url if config.server else None,
    }

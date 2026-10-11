"""Configuration loading.

API keys are referenced indirectly: configuration holds the name of an
environment variable, never the secret itself, and nothing here ever prints or
returns a key value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ServerConfig:
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 300.0


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    type: str
    model: str
    transport: str = "in_process"
    server: ServerConfig | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def credential_configured(self) -> bool:
        """Whether a required API key is present, without revealing it.

        A loopback server needs no key, so it counts as configured.
        """
        if self.transport != "openai_http":
            return True
        if self.server is None or not self.server.api_key_env:
            return True
        if self.server.base_url and _is_loopback(self.server.base_url):
            return True
        return bool(os.environ.get(self.server.api_key_env))

    def api_key(self) -> str | None:
        """Resolve the key at call time so it never lives in a config object."""
        if self.server is None or not self.server.api_key_env:
            return None
        return os.environ.get(self.server.api_key_env)


def _is_loopback(url: str) -> bool:
    host = url.split("//")[-1].split("/")[0].split(":")[0].strip("[]")
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


@dataclass(frozen=True)
class RunDefaults:
    word_timestamps: bool = True
    target_sample_rate: int = 16000
    target_channels: int = 1
    window_seconds: float = 28.0
    window_overlap_seconds: float = 1.0


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    audio_dir: Path
    results_dir: Path
    db_path: Path
    providers_yaml: Path
    providers: dict[str, ProviderConfig]
    defaults: RunDefaults

    def provider(self, name: str) -> ProviderConfig:
        try:
            return self.providers[name]
        except KeyError:
            known = ", ".join(sorted(self.providers)) or "none"
            raise KeyError(f"unknown provider {name!r} (known: {known})") from None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"provider configuration not found at {path}. "
            "Expected a providers.yaml at the project root."
        )
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings(root: Path | None = None, providers_yaml: Path | None = None) -> Settings:
    root = (root or ROOT).resolve()
    providers_yaml = providers_yaml or (root / "providers.yaml")
    raw = _load_yaml(providers_yaml)

    providers: dict[str, ProviderConfig] = {}
    for name, spec in (raw.get("providers") or {}).items():
        server_spec = spec.get("server") or None
        server = ServerConfig(**server_spec) if server_spec else None
        known = {"type", "model", "transport", "server"}
        providers[name] = ProviderConfig(
            name=name,
            type=spec["type"],
            model=spec.get("model", name),
            transport=spec.get("transport", "in_process"),
            server=server,
            options={k: v for k, v in spec.items() if k not in known},
        )

    defaults = RunDefaults(**(raw.get("defaults") or {}))

    data_dir = root / "data"
    return Settings(
        root=root,
        data_dir=data_dir,
        audio_dir=data_dir / "audio",
        results_dir=data_dir / "results",
        db_path=data_dir / "bench.sqlite3",
        providers_yaml=providers_yaml,
        providers=providers,
        defaults=defaults,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()

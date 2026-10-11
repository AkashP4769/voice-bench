"""Environment facts for the reproducibility record.

A benchmark number is only meaningful alongside what produced it. These are the
facts that can be gathered cheaply and portably; provider-specific detail (such
as Phonon's CPU instruction tier) is collected separately because it costs a
process start.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGES = (
    "cactus-needle",
    "fermion-research",
    "torch",
    "numpy",
    "soundfile",
    "soxr",
)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect() -> dict[str, Any]:
    """Machine and library facts, with nothing machine-identifying."""
    import os

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }


def phonon_cpu_tier(executable: str | None = None, timeout_s: float = 60.0) -> dict[str, Any] | None:
    """Ask the Fermion CLI which kernel tier this CPU selects.

    Best effort: the command loads a model, so it is only run when explicitly
    requested, and a failure returns ``None`` rather than breaking a run.
    """
    command = [executable or sys.executable, "-m", "fermion", "describe", "--json"]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    import json

    for line in reversed((completed.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def write_report(path: Path) -> dict[str, Any]:
    report = collect()
    report["phonon"] = phonon_cpu_tier()
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

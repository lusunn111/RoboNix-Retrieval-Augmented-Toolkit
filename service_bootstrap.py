"""Runtime helpers for the import-compatible RT-Cache source snapshot."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, Dict

SERVICE_ROOT = Path(__file__).resolve().parent
RTCACHE_ROOT = SERVICE_ROOT / "vendor" / "rtcache"


def activate_vendor() -> Path:
    """Expose RT-Cache without importing models or connecting to databases."""
    value = str(RTCACHE_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    return RTCACHE_ROOT


def run_rtcache_script(relative_path: str) -> Dict[str, Any]:
    """Run an original RT-Cache script with its CLI arguments unchanged."""
    root = activate_vendor()
    script = (root / relative_path).resolve()
    if root.resolve() not in script.parents or not script.is_file():
        raise FileNotFoundError(f"Unknown RT-Cache script: {relative_path}")
    return runpy.run_path(str(script), run_name="__main__")

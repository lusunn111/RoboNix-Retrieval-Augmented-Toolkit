"""HeiSD retrieval verification and execution implementation locations."""

from pathlib import Path

try:
    from retrieval_augmented_service.service_bootstrap import SERVICE_ROOT
except ModuleNotFoundError as exc:
    if exc.name != "retrieval_augmented_service":
        raise
    from service_bootstrap import SERVICE_ROOT

SPECVLA_ROOT = SERVICE_ROOT / "benchmarks" / "specvla_validation"
REBUTTAL_ROOT = SERVICE_ROOT / "benchmarks" / "rebuttal"


def implementation_path(name: str = "specvla") -> Path:
    paths = {"specvla": SPECVLA_ROOT, "rebuttal": REBUTTAL_ROOT}
    try:
        return paths[name]
    except KeyError as exc:
        raise KeyError(f"Unknown verified-execution implementation: {name}") from exc


__all__ = ["SPECVLA_ROOT", "REBUTTAL_ROOT", "implementation_path"]

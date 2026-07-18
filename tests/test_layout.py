from pathlib import Path
import subprocess
import sys

try:
    import retrieval_augmented_service as service
    from retrieval_augmented_service.modules.verified_execution import implementation_path

    SERVICE_ROOT = Path(service.__file__).resolve().parent
    activate_vendor = service.activate_vendor
except ModuleNotFoundError:
    import service_bootstrap
    from modules.verified_execution import implementation_path

    SERVICE_ROOT = Path(service_bootstrap.__file__).resolve().parent
    activate_vendor = service_bootstrap.activate_vendor


def test_canonical_source_trees_exist():
    assert (SERVICE_ROOT / "vendor" / "rtcache" / "scripts").is_dir()
    assert implementation_path("specvla").is_dir()
    assert implementation_path("rebuttal").is_dir()


def test_activate_vendor_is_idempotent():
    assert activate_vendor() == activate_vendor()


def test_cli_help_works_from_independent_toolkit_root():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run", "--help"],
        cwd=SERVICE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Path relative to vendor/rtcache" in result.stdout

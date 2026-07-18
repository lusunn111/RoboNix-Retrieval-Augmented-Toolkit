"""Run an original RT-Cache script from the compatibility snapshot."""

from __future__ import annotations

import argparse
import sys

try:
    from retrieval_augmented_service.service_bootstrap import run_rtcache_script
except ModuleNotFoundError as exc:
    if exc.name != "retrieval_augmented_service":
        raise
    # Support an independently extracted toolkit whose current directory is
    # retrieval_augmented_service/ rather than its parent monorepo.
    from service_bootstrap import run_rtcache_script


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="Path relative to vendor/rtcache")
    args, forwarded = parser.parse_known_args()
    sys.argv = [args.script, *forwarded]
    run_rtcache_script(args.script)


if __name__ == "__main__":
    main()

from __future__ import annotations

import platform
import sys


def worker_platform_is_supported(
    *,
    python_version: tuple[int, int, int] | None = None,
    system: str | None = None,
    machine: str | None = None,
    macos_version: str | None = None,
) -> bool:
    version = python_version or tuple(sys.version_info[:3])
    current_system = system or platform.system()
    current_machine = (machine or platform.machine()).casefold()
    current_macos = macos_version if macos_version is not None else platform.mac_ver()[0]
    try:
        macos_major = int(current_macos.split(".", 1)[0])
    except (TypeError, ValueError):
        return False
    return (
        version == (3, 12, 13)
        and current_system == "Darwin"
        and current_machine in {"arm64", "aarch64"}
        and macos_major >= 14
    )


def main() -> None:
    if not worker_platform_is_supported():
        raise SystemExit(
            "Isolated worker locks and runtimes require Python 3.12.13 "
            "on Apple Silicon with macOS 14 or newer"
        )


if __name__ == "__main__":
    main()

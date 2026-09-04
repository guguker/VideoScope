from __future__ import annotations

import argparse
import platform
import sys


def worker_host_is_supported(
    *,
    system: str | None = None,
    machine: str | None = None,
    macos_version: str | None = None,
) -> bool:
    current_system = system or platform.system()
    current_machine = (machine or platform.machine()).casefold()
    current_macos = macos_version if macos_version is not None else platform.mac_ver()[0]
    try:
        macos_major = int(current_macos.split(".", 1)[0])
    except (TypeError, ValueError):
        return False
    return (
        current_system == "Darwin"
        and current_machine in {"arm64", "aarch64"}
        and macos_major >= 14
    )


def worker_platform_is_supported(
    *,
    python_version: tuple[int, int, int] | None = None,
    expected_python_version: tuple[int, int, int] = (3, 12, 13),
    system: str | None = None,
    machine: str | None = None,
    macos_version: str | None = None,
) -> bool:
    version = python_version or tuple(sys.version_info[:3])
    return (
        version == expected_python_version
        and worker_host_is_supported(
            system=system,
            machine=machine,
            macos_version=macos_version,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-only", action="store_true")
    parser.add_argument("--python-version", default="3.12.13")
    arguments = parser.parse_args()
    try:
        expected_python_version = tuple(
            int(part) for part in arguments.python_version.split(".")
        )
    except ValueError as error:
        raise SystemExit("Expected Python version must be MAJOR.MINOR.PATCH") from error
    if len(expected_python_version) != 3:
        raise SystemExit("Expected Python version must be MAJOR.MINOR.PATCH")
    supported = (
        worker_host_is_supported()
        if arguments.host_only
        else worker_platform_is_supported(
            expected_python_version=expected_python_version,
        )
    )
    if not supported:
        raise SystemExit(
            "Isolated workers require Apple Silicon with macOS 14 or newer"
            if arguments.host_only
            else "Isolated worker locks and runtimes require Python "
            f"{arguments.python_version} on Apple Silicon with macOS 14 or newer"
        )


if __name__ == "__main__":
    main()

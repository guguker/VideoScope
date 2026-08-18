from __future__ import annotations


_DEPRECATION_MESSAGE = (
    "This legacy script is disabled because it cannot publish an immutable "
    "text-vector generation. Start VideoScope and use "
    "POST /api/videos/{video_id}/reindex instead."
)


def main() -> None:
    raise SystemExit(_DEPRECATION_MESSAGE)


if __name__ == "__main__":
    main()

import uvicorn

from videoscope.config import AppSettings


def main() -> None:
    settings = AppSettings()
    uvicorn.run(
        "videoscope.api:create_app",
        host=settings.host,
        port=settings.port,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()


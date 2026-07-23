from videoscope.providers.base import ProviderRegistry, ProviderState, StaticProvider


def test_registry_reports_ready_optional_and_unavailable_providers() -> None:
    registry = ProviderRegistry(
        [
            StaticProvider("ffmpeg", "FFmpeg", ProviderState.READY, "8.1"),
            StaticProvider(
                "roboflow",
                "Roboflow",
                ProviderState.NEEDS_CONFIGURATION,
                "API key is missing",
                optional=True,
            ),
        ]
    )

    statuses = registry.statuses()

    assert statuses[0].id == "ffmpeg"
    assert statuses[0].state is ProviderState.READY
    assert statuses[1].optional is True
    assert registry.is_ready("roboflow") is False


def test_registry_rejects_duplicate_provider_ids() -> None:
    provider = StaticProvider("ffmpeg", "FFmpeg", ProviderState.READY, "ok")

    try:
        ProviderRegistry([provider, provider])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate provider ID should fail")


from pathlib import Path

from videoscope.config import AppSettings
from videoscope.providers.base import ProviderState
from videoscope.providers.lighthouse import DisabledLighthouseRetriever, LighthouseRetriever
from videoscope.providers.lighthouse_worker import LighthouseWorkerClient
from videoscope.runtime import create_lighthouse_retriever


class FakeFFmpeg:
    pass


def test_runtime_builds_worker_adapter_without_importing_model_stack(tmp_path: Path) -> None:
    settings = AppSettings(
        data_dir=tmp_path,
        lighthouse_endpoint="http://127.0.0.1:8782",
        lighthouse_api_key="l" * 32,
        lighthouse_timeout=47,
    )

    retriever = create_lighthouse_retriever(settings, FakeFFmpeg())  # type: ignore[arg-type]

    assert isinstance(retriever, LighthouseWorkerClient)
    assert retriever.timeout == 47
    assert retriever.input_root == settings.media_dir.resolve()
    assert retriever.identity["mode"] == "isolated-worker"


def test_runtime_lighthouse_is_disabled_without_explicit_boundary(tmp_path: Path) -> None:
    retriever = create_lighthouse_retriever(
        AppSettings(data_dir=tmp_path),
        FakeFFmpeg(),  # type: ignore[arg-type]
    )

    assert isinstance(retriever, DisabledLighthouseRetriever)
    assert retriever.status().state is ProviderState.NEEDS_CONFIGURATION
    assert retriever.search("query", ["video-1"]) == []
    assert retriever.cache_is_current("video-1") is False


def test_deprecated_in_process_compatibility_is_explicit(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path, lighthouse_allow_in_process=True)

    retriever = create_lighthouse_retriever(settings, FakeFFmpeg())  # type: ignore[arg-type]

    assert isinstance(retriever, LighthouseRetriever)
    assert retriever.identity["mode"] == "deprecated-in-process"

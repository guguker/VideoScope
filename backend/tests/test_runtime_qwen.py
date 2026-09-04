from pathlib import Path

from videoscope.config import AppSettings
from videoscope.model_manifest import QWEN_VIDEO_MODEL
from videoscope.providers.qwen_worker import QwenWorkerClient
from videoscope.providers.qwen_video import QWEN_IN_PROCESS_RUNTIME_IDENTITY
from videoscope.runtime import create_qwen_reranker


class UnusedRepository:
    pass


class UnusedExtractor:
    pass


def test_runtime_builds_worker_adapter_without_importing_or_contacting_mlx(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        data_dir=tmp_path,
        qwen_video_model=QWEN_VIDEO_MODEL,
        qwen_video_endpoint="http://127.0.0.1:8781",
        qwen_video_api_key="q" * 32,
        qwen_video_timeout=41,
    )
    settings.temp_dir.mkdir(parents=True)

    reranker = create_qwen_reranker(
        settings,
        UnusedRepository(),  # type: ignore[arg-type]
        UnusedExtractor(),  # type: ignore[arg-type]
    )

    assert isinstance(reranker.inference_client, QwenWorkerClient)
    assert reranker.allow_in_process is False
    assert reranker.inference_client.timeout == 41
    assert reranker.inference_client.input_root == settings.temp_dir.resolve()
    assert reranker.identity["boundary"]["mode"] == "isolated-worker"  # type: ignore[index]


def test_runtime_keeps_deprecated_in_process_path_behind_explicit_flag(
    tmp_path: Path,
) -> None:
    settings = AppSettings(
        data_dir=tmp_path,
        qwen_video_model=QWEN_VIDEO_MODEL,
        qwen_video_allow_in_process=True,
    )

    reranker = create_qwen_reranker(
        settings,
        UnusedRepository(),  # type: ignore[arg-type]
        UnusedExtractor(),  # type: ignore[arg-type]
    )

    assert reranker.inference_client is None
    assert reranker.allow_in_process is True
    assert reranker.identity["boundary"] == {
        "mode": "deprecated-in-process",
        "runtime_identity": QWEN_IN_PROCESS_RUNTIME_IDENTITY,
    }

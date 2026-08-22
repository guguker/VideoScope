#!/usr/bin/env python3
"""Run the browser smoke-test backend without production models or user data."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import uvicorn

from videoscope.api import create_app
from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    StageState,
)
from videoscope.config import AppSettings
from videoscope.jobs import VideoIndexPlanSnapshot
from videoscope.processing.dispatcher import DurableVideoIndexDispatcher
from videoscope.processing.indexer import VideoIndexExecutionContext
from videoscope.providers.base import ProviderRegistry, ProviderState, StaticProvider
from videoscope.providers.whisper import WhisperPromptSnapshot
from videoscope.repository import Repository


_EXECUTOR_IDENTITY = "sha256:" + hashlib.sha256(
    b"videoscope-deterministic-e2e-executor-v1"
).hexdigest()
_PROCESSING_WINDOW_SECONDS = 2.25


def _stage(kind: StageKind) -> StageSpecification:
    parameters: dict[str, object] = {}
    if kind is StageKind.SPEECH:
        parameters = {
            "effective_prompt_sha256": hashlib.sha256(b"").hexdigest(),
            "glossary_state": "not_configured",
        }
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"videoscope.e2e.{kind.value}.v1",
        parameters=parameters,
        dependencies={"harness": "deterministic-v1"},
    )


def _plan() -> VideoIndexPlanSnapshot:
    prompt = WhisperPromptSnapshot(
        effective_prompt=None,
        effective_prompt_sha256=hashlib.sha256(b"").hexdigest(),
        glossary_state="not_configured",
    )
    return VideoIndexPlanSnapshot(
        schema_version=2,
        specifications=IndexingSpecifications(
            scenes=_stage(StageKind.SCENES),
            speech=_stage(StageKind.SPEECH),
            ocr=_stage(StageKind.OCR),
            objects=_stage(StageKind.OBJECTS),
            text_vectors=_stage(StageKind.TEXT_VECTORS),
        ),
        visual_dense_specification=_stage(StageKind.VISUAL_DENSE),
        lighthouse_specification=_stage(StageKind.LIGHTHOUSE),
        whisper_prompt_snapshot=prompt,
        executor_identity=_EXECUTOR_IDENTITY,
    )


class _DeterministicVideoIndexExecutor:
    def __init__(self, repository: Repository, *, media_root: Path) -> None:
        self.repository = repository
        self.media_root = media_root

    def process_durable(
        self,
        video_id: str,
        *,
        context: VideoIndexExecutionContext,
    ) -> tuple[str, ...]:
        job = self.repository.get_video_index_job(context.job_id)
        if job is None or job.video_id != video_id:
            raise RuntimeError("deterministic job binding is unavailable")
        self.repository.verify_existing_asset_identity(
            video_id,
            media_root=self.media_root,
        )
        self.repository.checkpoint_video_index_job(
            context.job_id,
            execution_token=context.execution_token,
            progress=0.25,
            stage="scenes",
        )
        Event().wait(timeout=_PROCESSING_WINDOW_SECONDS)
        specifications = (
            *context.plan.specifications.segment_specifications,
            context.plan.specifications.text_vectors,
            context.plan.visual_dense_specification,
            context.plan.lighthouse_specification,
        )
        for specification in specifications:
            run = self.repository.create_stage_run(
                video_id=video_id,
                specification=specification,
                run_id=f"{context.job_id}-{specification.kind.value}",
                job_id=context.job_id,
                execution_token=context.execution_token,
            )
            self.repository.transition_stage_run(
                run.run_id,
                StageState.NOT_CONFIGURED,
                execution_token=context.execution_token,
            )
        self.repository.checkpoint_video_index_job(
            context.job_id,
            execution_token=context.execution_token,
            progress=0.95,
            stage="finalizing",
        )
        return ()


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1024 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1024 and 65535")
    return port


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    parser.add_argument("--port", type=_port, default=8876)
    arguments = parser.parse_args()

    with TemporaryDirectory(prefix="videoscope-e2e-") as temporary:
        temporary_root = Path(temporary).resolve(strict=True)
        settings = AppSettings(
            _env_file=None,
            data_dir=temporary_root / "data",
            host=arguments.host,
            port=arguments.port,
            max_upload_bytes=1024 * 1024,
        )
        settings.ensure_directories()
        repository = Repository(settings.database_path)
        repository.initialize()
        dispatcher = DurableVideoIndexDispatcher(
            repository,
            _DeterministicVideoIndexExecutor(
                repository,
                media_root=settings.media_dir,
            ),
            executor_identity_resolver=lambda: _EXECUTOR_IDENTITY,
            poll_interval=0.05,
            start_immediately=False,
        )
        dispatcher.recover_startup()
        dispatcher.start()
        app = create_app(
            settings=settings,
            repository=repository,
            processing_queue=dispatcher,
            provider_registry=ProviderRegistry(
                [
                    StaticProvider(
                        "deterministic-e2e",
                        "Deterministic E2E",
                        ProviderState.READY,
                        "test-only executor",
                    )
                ]
            ),
            video_index_plan_factory=_plan,
        )
        try:
            uvicorn.run(
                app,
                host=arguments.host,
                port=arguments.port,
                access_log=False,
                log_level="warning",
            )
        finally:
            dispatcher.close(timeout=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

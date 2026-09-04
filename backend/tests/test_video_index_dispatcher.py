from __future__ import annotations

from threading import Event
from time import monotonic, sleep

import pytest

from videoscope.artifacts import (
    IndexingSpecifications,
    StageKind,
    StageSpecification,
    StageState,
)
from videoscope.jobs import JobState, VideoIndexPlanSnapshot
import videoscope.processing.dispatcher as dispatcher_module
from videoscope.processing.dispatcher import DurableVideoIndexDispatcher
from videoscope.processing.indexer import JobCancelled, VideoIndexExecutionContext
from videoscope.providers.whisper import WhisperPromptSnapshot
from videoscope.repository import Repository


TOKEN = "dispatcher-token-aaaaaaaaaaaaaaaa"


def _stage(kind: StageKind) -> StageSpecification:
    parameters: dict[str, object] = {}
    if kind is StageKind.SPEECH:
        parameters = {
            "effective_prompt_sha256": "e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855",
            "glossary_state": "not_configured",
        }
    return StageSpecification(
        kind=kind,
        schema_version=1,
        implementation_revision=f"test.{kind.value}.v1",
        parameters=parameters,
        dependencies={"videoscope": "test"},
    )


def _plan(*, executor: str = "a") -> VideoIndexPlanSnapshot:
    prompt = WhisperPromptSnapshot(
        effective_prompt=None,
        effective_prompt_sha256="e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855",
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
        executor_identity="sha256:" + executor * 64,
    )


def _repository(tmp_path) -> Repository:  # type: ignore[no-untyped-def]
    repository = Repository(tmp_path / "videoscope.sqlite3")
    repository.initialize()
    return repository


def _enqueue(
    repository: Repository,
    *,
    video_id: str,
    job_id: str,
    plan: VideoIndexPlanSnapshot,
) -> None:
    repository.create_video_with_asset_and_index_job(
        video_id=video_id,
        original_name=f"{video_id}.mp4",
        stored_name=f"{video_id}.mp4",
        media_path=f"/managed/{video_id}.mp4",
        size_bytes=1024,
        source_sha256=("a" if video_id.endswith("1") else "b") * 64,
        plan=plan,
        job_id=job_id,
    )


def _record_terminal_receipts(
    repository: Repository,
    video_id: str,
    context: VideoIndexExecutionContext,
) -> None:
    specifications = (
        *context.plan.specifications.segment_specifications,
        context.plan.specifications.text_vectors,
        context.plan.visual_dense_specification,
        context.plan.lighthouse_specification,
    )
    for specification in specifications:
        run = repository.create_stage_run(
            video_id=video_id,
            specification=specification,
            run_id=f"{context.job_id}-{specification.kind.value}",
            job_id=context.job_id,
            execution_token=context.execution_token,
        )
        repository.transition_stage_run(
            run.run_id,
            StageState.NOT_CONFIGURED,
            execution_token=context.execution_token,
        )


def _wait_for_state(
    repository: Repository,
    job_id: str,
    state: JobState,
    *,
    timeout: float = 5.0,
):  # type: ignore[no-untyped-def]
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        job = repository.get_video_index_job(job_id)
        if job is not None and job.state is state:
            return job
        sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {state.value}")


def test_dispatcher_claims_preexisting_durable_work_without_submit(tmp_path) -> None:
    repository = _repository(tmp_path)
    plan = _plan()
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)
    processed = Event()
    contexts: list[VideoIndexExecutionContext] = []

    class Executor:
        def process_durable(
            self,
            video_id: str,
            *,
            context: VideoIndexExecutionContext,
        ) -> tuple[str, ...]:
            assert video_id == "video-1"
            contexts.append(context)
            _record_terminal_receipts(repository, video_id, context)
            processed.set()
            return ()

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: plan.executor_identity,
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    try:
        dispatcher.start()
        assert processed.wait(timeout=2)
        completed = _wait_for_state(repository, "job-1", JobState.COMPLETE)
    finally:
        assert dispatcher.close(timeout=2)

    assert completed.execution_token is None
    assert len(contexts) == 1
    assert contexts[0].plan.canonical_json == plan.canonical_json
    assert contexts[0].execution_token == TOKEN


def test_dispatcher_fails_closed_when_executor_returns_without_stage_receipts(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    plan = _plan()
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)

    class NoOpExecutor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return ()

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        NoOpExecutor(),
        executor_identity_resolver=lambda: plan.executor_identity,
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    try:
        dispatcher.start()
        failed = _wait_for_state(repository, "job-1", JobState.FAILED)
    finally:
        assert dispatcher.close(timeout=2)

    assert failed.error_code == "job_execution_failed"


def test_dispatcher_fails_closed_before_execution_on_executor_identity_drift(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    plan = _plan(executor="a")
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)

    class Executor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("executor must not run under another identity")

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: "sha256:" + "b" * 64,
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    try:
        dispatcher.start()
        failed = _wait_for_state(repository, "job-1", JobState.FAILED)
    finally:
        assert dispatcher.close(timeout=2)

    assert failed.error_code == "job_executor_identity_mismatch"


def test_dispatcher_rechecks_executor_identity_after_work_before_publication(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    plan = _plan(executor="a")
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)
    current_identity = [plan.executor_identity]

    class Executor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            current_identity[0] = "sha256:" + "b" * 64
            return ()

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: current_identity[0],
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    try:
        dispatcher.start()
        failed = _wait_for_state(repository, "job-1", JobState.FAILED)
    finally:
        assert dispatcher.close(timeout=2)

    assert failed.error_code == "job_executor_identity_mismatch"


def test_dispatcher_persists_only_a_sanitized_execution_failure(tmp_path) -> None:
    repository = _repository(tmp_path)
    plan = _plan()
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)

    class Executor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("secret filesystem path /private/model")

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: plan.executor_identity,
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    try:
        dispatcher.start()
        failed = _wait_for_state(repository, "job-1", JobState.FAILED)
    finally:
        assert dispatcher.close(timeout=2)

    assert failed.error_code == "job_execution_failed"
    assert "/private/model" not in repr(failed)


def test_dispatcher_acknowledges_a_running_cancellation(tmp_path) -> None:
    repository = _repository(tmp_path)
    plan = _plan()
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)
    entered = Event()
    release = Event()

    class Executor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            entered.set()
            assert release.wait(timeout=2)
            raise JobCancelled("requested")

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: plan.executor_identity,
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    try:
        dispatcher.start()
        assert entered.wait(timeout=2)
        requested = repository.request_video_index_job_cancellation("job-1")
        assert requested.cancel_requested_at is not None
        release.set()
        cancelled = _wait_for_state(repository, "job-1", JobState.CANCELLED)
    finally:
        release.set()
        assert dispatcher.close(timeout=2)

    assert cancelled.error_code is None


def test_dispatcher_shutdown_finishes_owned_work_but_preserves_db_backlog(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    plan = _plan()
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)
    _enqueue(repository, video_id="video-2", job_id="job-2", plan=plan)
    entered = Event()
    release = Event()
    calls: list[str] = []

    class Executor:
        def process_durable(
            self,
            video_id: str,
            *,
            context: VideoIndexExecutionContext,
        ) -> tuple[str, ...]:
            calls.append(video_id)
            entered.set()
            assert release.wait(timeout=2)
            _record_terminal_receipts(repository, video_id, context)
            return ()

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: plan.executor_identity,
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    dispatcher.start()
    assert entered.wait(timeout=2)
    assert dispatcher.close(timeout=0.01) is False
    release.set()
    assert dispatcher.close(timeout=2) is True

    assert _wait_for_state(repository, "job-1", JobState.COMPLETE)
    queued = repository.get_video_index_job("job-2")
    assert queued is not None and queued.state is JobState.QUEUED
    assert calls == ["video-1"]


def test_dispatcher_recovery_terminalizes_abandoned_attempts_before_start(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    plan = _plan()
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)
    repository.claim_next_video_index_job(execution_token=TOKEN)

    class Executor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("recovery does not execute work")

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: plan.executor_identity,
        start_immediately=False,
    )
    report = dispatcher.recover_startup()

    parent = repository.get_video_index_job("job-1")
    assert parent is not None and parent.state is JobState.FAILED
    assert parent.error_code == "job_runtime_abandoned"
    assert report.retry_job_ids
    child = repository.get_video_index_job(report.retry_job_ids[0])
    assert child is not None and child.state is JobState.QUEUED
    assert dispatcher.close()


def test_dispatcher_lifecycle_is_idempotent_and_rejects_restart(tmp_path) -> None:
    repository = _repository(tmp_path)

    class Executor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return ()

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: "sha256:" + "a" * 64,
        start_immediately=False,
        poll_interval=0.01,
    )
    dispatcher.start()
    with pytest.raises(RuntimeError, match="already started"):
        dispatcher.start()
    assert dispatcher.close(timeout=2)
    assert dispatcher.close(timeout=2)
    with pytest.raises(RuntimeError, match="closed"):
        dispatcher.start()


@pytest.mark.parametrize("poll_interval", [True, 0.0, 60.01, "fast"])
def test_dispatcher_rejects_invalid_poll_intervals(tmp_path, poll_interval) -> None:  # type: ignore[no-untyped-def]
    repository = _repository(tmp_path)

    class Executor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return ()

    with pytest.raises(ValueError, match="poll interval"):
        DurableVideoIndexDispatcher(
            repository,
            Executor(),
            executor_identity_resolver=lambda: "sha256:" + "a" * 64,
            poll_interval=poll_interval,
            start_immediately=False,
        )


def test_dispatcher_requires_its_exact_repository_and_executor_contract(tmp_path) -> None:
    repository = _repository(tmp_path)

    with pytest.raises(TypeError, match="repository"):
        DurableVideoIndexDispatcher(  # type: ignore[arg-type]
            object(),
            object(),
            executor_identity_resolver=lambda: "sha256:" + "a" * 64,
        )
    with pytest.raises(TypeError, match="indexer"):
        DurableVideoIndexDispatcher(
            repository,
            object(),  # type: ignore[arg-type]
            executor_identity_resolver=lambda: "sha256:" + "a" * 64,
        )
    with pytest.raises(TypeError, match="identity resolver"):
        DurableVideoIndexDispatcher(
            repository,
            SimpleExecutor(),
            executor_identity_resolver=None,  # type: ignore[arg-type]
        )


def test_default_execution_token_uses_exact_256_bit_hex(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_sizes: list[int] = []
    monkeypatch.setattr(
        dispatcher_module.secrets,
        "token_urlsafe",
        lambda _size: "_" + "a" * 42,
    )
    monkeypatch.setattr(
        dispatcher_module.secrets,
        "token_hex",
        lambda size: generated_sizes.append(size) or "ab" * size,
    )
    dispatcher = DurableVideoIndexDispatcher(
        _repository(tmp_path),
        SimpleExecutor(),
        executor_identity_resolver=lambda: "sha256:" + "a" * 64,
        start_immediately=False,
    )

    assert dispatcher._new_execution_token() == "ab" * 32
    assert generated_sizes == [32]


@pytest.mark.parametrize(
    "token",
    [
        "_" + "a" * 63,
        "-" + "a" * 63,
        "short",
        "a" * 15 + "!",
        "a" * 257,
        7,
    ],
)
def test_dispatcher_rejects_invalid_factory_token_before_claim(
    tmp_path,
    token: object,
) -> None:
    dispatcher = DurableVideoIndexDispatcher(
        _repository(tmp_path),
        SimpleExecutor(),
        executor_identity_resolver=lambda: "sha256:" + "a" * 64,
        token_factory=lambda: token,  # type: ignore[return-value]
        start_immediately=False,
    )

    with pytest.raises(ValueError, match="token factory"):
        dispatcher._new_execution_token()


class SimpleExecutor:
    def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return ()


def test_dispatcher_fails_closed_when_current_executor_cannot_be_attested(
    tmp_path,
) -> None:
    repository = _repository(tmp_path)
    plan = _plan()
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)

    def unavailable() -> str:
        raise RuntimeError("local detail")

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        SimpleExecutor(),
        executor_identity_resolver=unavailable,
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    try:
        dispatcher.start()
        failed = _wait_for_state(repository, "job-1", JobState.FAILED)
    finally:
        assert dispatcher.close(timeout=2)
    assert failed.error_code == "job_executor_unavailable"


def test_cancel_racing_success_wins_before_job_publication(tmp_path) -> None:
    repository = _repository(tmp_path)
    plan = _plan()
    _enqueue(repository, video_id="video-1", job_id="job-1", plan=plan)

    class Executor:
        def process_durable(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            repository.request_video_index_job_cancellation("job-1")
            return ()

    dispatcher = DurableVideoIndexDispatcher(
        repository,
        Executor(),
        executor_identity_resolver=lambda: plan.executor_identity,
        poll_interval=0.01,
        token_factory=lambda: TOKEN,
        start_immediately=False,
    )
    try:
        dispatcher.start()
        cancelled = _wait_for_state(repository, "job-1", JobState.CANCELLED)
    finally:
        assert dispatcher.close(timeout=2)
    assert cancelled.finished_at is not None


def test_dispatcher_wake_is_advisory_and_rejected_after_close(tmp_path) -> None:
    dispatcher = DurableVideoIndexDispatcher(
        _repository(tmp_path),
        SimpleExecutor(),
        executor_identity_resolver=lambda: "sha256:" + "a" * 64,
        start_immediately=False,
    )
    dispatcher.wake()
    assert dispatcher.close()
    with pytest.raises(RuntimeError, match="closed"):
        dispatcher.wake()
    with pytest.raises(RuntimeError, match="closed"):
        dispatcher.recover_startup()


def test_dispatcher_thread_start_failure_remains_safely_closeable(
    tmp_path,
    monkeypatch,
) -> None:
    class FailingThread:
        def __init__(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

        def start(self) -> None:
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(dispatcher_module, "Thread", FailingThread)
    dispatcher = DurableVideoIndexDispatcher(
        _repository(tmp_path),
        SimpleExecutor(),
        executor_identity_resolver=lambda: "sha256:" + "a" * 64,
        start_immediately=False,
    )
    with pytest.raises(RuntimeError, match="thread unavailable"):
        dispatcher.start()
    assert dispatcher.close()

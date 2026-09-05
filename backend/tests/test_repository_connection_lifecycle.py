from __future__ import annotations

from contextlib import closing
import gc
from hashlib import sha256
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
import pytest

import videoscope.repository as repository_module
from videoscope.repository import Repository


def _assert_closed(connection: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")


def _stored_values(database_path: Path) -> list[int]:
    with closing(sqlite3.connect(database_path)) as connection:
        return [row[0] for row in connection.execute("SELECT value FROM samples")]


def test_connection_context_commits_before_closing(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "library.sqlite3")
    with repository._connect() as connection:
        connection.execute("CREATE TABLE samples (value INTEGER)")
        connection.execute("INSERT INTO samples VALUES (7)")
    try:
        _assert_closed(connection)
        assert _stored_values(repository.database_path) == [7]
    finally:
        connection.close()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_connection_context_rolls_back_then_closes_on_exception(
    tmp_path: Path,
    error_type: type[BaseException],
) -> None:
    repository = Repository(tmp_path / "library.sqlite3")
    with repository._connect() as setup:
        setup.execute("CREATE TABLE samples (value INTEGER)")
    failure = error_type("transaction interrupted")
    with pytest.raises(error_type) as captured:
        with repository._connect() as connection:
            connection.execute("INSERT INTO samples VALUES (7)")
            raise failure
    try:
        assert captured.value is failure
        _assert_closed(connection)
        assert _stored_values(repository.database_path) == []
    finally:
        connection.close()
        setup.close()


def test_connection_context_closes_after_failed_commit(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "library.sqlite3")
    with repository._connect() as setup:
        setup.execute("CREATE TABLE parents (value INTEGER PRIMARY KEY)")
        setup.execute(
            "CREATE TABLE samples (value INTEGER REFERENCES parents(value) "
            "DEFERRABLE INITIALLY DEFERRED)"
        )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with repository._connect() as connection:
            connection.execute("INSERT INTO samples VALUES (7)")
    try:
        _assert_closed(connection)
        assert _stored_values(repository.database_path) == []
    finally:
        connection.close()
        setup.close()


def test_explicit_connection_remains_owned_until_caller_closes(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "library.sqlite3")
    connection = repository._connect()
    assert isinstance(connection, sqlite3.Connection)
    try:
        connection.execute("CREATE TABLE samples (value INTEGER)")
        connection.execute("INSERT INTO samples VALUES (1)")
        connection.commit()
        connection.execute("INSERT INTO samples VALUES (2)")
        connection.rollback()
        assert [row[0] for row in connection.execute("SELECT value FROM samples")] == [1]
    finally:
        connection.close()
    _assert_closed(connection)
    connection.close()


def test_read_only_connection_context_closes_without_losing_query_only(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "library.sqlite3")
    repository.initialize()
    read_only = Repository.open_read_only(repository.database_path)
    with read_only._connect() as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE forbidden (value INTEGER)")
    try:
        _assert_closed(connection)
    finally:
        connection.close()


@pytest.mark.parametrize("setup_step", ["row_factory", "foreign_keys", "query_only"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_connection_setup_failure_closes_new_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup_step: str,
    error_type: type[BaseException],
) -> None:
    failure = error_type("connection setup failed")

    class FailingConnection:
        closed = False

        def __setattr__(self, name: str, value: object) -> None:
            if name == "row_factory" and setup_step == "row_factory":
                raise failure
            object.__setattr__(self, name, value)

        def execute(self, statement: str) -> None:
            if setup_step in statement:
                raise failure

        def close(self) -> None:
            self.closed = True

    connection = FailingConnection()
    repository = Repository(tmp_path / "library.sqlite3")
    if setup_step == "query_only":
        repository._read_only = True
        repository._read_only_uri = repository.database_path.as_uri() + "?mode=ro"
    monkeypatch.setattr(
        repository_module.sqlite3,
        "connect",
        lambda *_args, **_kwargs: connection,
    )
    with pytest.raises(error_type) as captured:
        repository._connect()
    assert captured.value is failure
    assert connection.closed is True


def test_completed_repository_context_cannot_mutate_qwen_ancestor_during_gc(
    tmp_path: Path,
) -> None:
    """A closed DB scope must not defer WAL unlinking into an unrelated request."""
    import videoscope.providers.qwen_worker as qwen_worker
    from videoscope.providers.qwen_video import QwenVideoJudgement

    product = tmp_path / "product"
    temporary = product / "tmp"
    temporary.mkdir(parents=True)
    source = temporary / "candidate.mp4"
    content = b"unchanged synthetic video"
    source.write_bytes(content)
    repository = Repository(product / "library.sqlite3")
    pending: list[sqlite3.Connection] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    observations: dict[str, object] = {}
    token = "q" * 32
    model = "fixture/qwen@" + "a" * 40

    def sidecars() -> list[str]:
        return sorted(path.name for path in product.glob("library.sqlite3-*"))

    class CollectingRuntime:
        model_identity = model
        loaded = True
        available = True

        def judge(self, _request: object, materialized: Path) -> QwenVideoJudgement:
            assert materialized != source
            assert materialized.read_bytes() == content
            observations["sidecars_before"] = sidecars()
            pending.clear()
            gc.collect()
            observations["sidecars_after"] = sidecars()
            assert source.read_bytes() == content
            return QwenVideoJudgement(matches_query=True, confidence=0.9)

    try:
        with repository._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("CREATE TABLE samples (value INTEGER)")
            connection.execute("INSERT INTO samples VALUES (1)")
        # Retained references are legal, but exiting the owning context must
        # already have closed its resources. Explicit GC models later collection.
        pending.append(connection)
        del connection
        before = source.stat()
        app = qwen_worker.create_qwen_worker_app(
            runtime=CollectingRuntime(),
            input_root=tmp_path,
            api_key=token,
        )
        request = {
            "schema_version": qwen_worker.QWEN_WORKER_SCHEMA_VERSION,
            "request_id": "a" * 32,
            "model_identity": model,
            "runtime_identity": qwen_worker.QWEN_INFERENCE_RUNTIME_IDENTITY,
            "source_bundle_sha256": qwen_worker.QWEN_SOURCE_BUNDLE_SHA256,
            "prompt_protocol_sha256": qwen_worker.QWEN_PROMPT_PROTOCOL_SHA256,
            "input_root_sha256": qwen_worker._input_root_identity(tmp_path),
            "input_kind": "video",
            "relative_path": "product/tmp/candidate.mp4",
            "expected_sha256": sha256(content).hexdigest(),
            "expected_byte_size": len(content),
            "prompt_kind": "basketball_facts",
            "query": None,
            "fps": 2.0,
            "max_tokens": 320,
        }
        with TestClient(
            app,
            base_url="http://127.0.0.1",
            client=("127.0.0.1", 50000),
        ) as client:
            response = client.post(
                "/v1/judge", json=request, headers={"Authorization": f"Bearer {token}"}
            )
        after = source.stat()
        assert source.read_bytes() == content
        assert (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
            before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
        )
        assert response.status_code == 200, (response.json(), observations)
        assert observations == {"sidecars_before": [], "sidecars_after": []}
    finally:
        for connection in pending:
            connection.close()
        pending.clear()
        gc.collect()
        if gc_was_enabled:
            gc.enable()

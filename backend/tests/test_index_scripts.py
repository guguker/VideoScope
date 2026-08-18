from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from videoscope.providers.base import ProviderState
from videoscope.runtime_lifecycle import ExclusiveRuntimeLock


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    script_path = PROJECT_ROOT / "scripts" / name
    specification = importlib.util.spec_from_file_location(
        f"videoscope_test_{name.replace('-', '_').removesuffix('.py')}",
        script_path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script_name", ["index-speech.py", "index-objects.py"])
def test_legacy_text_index_scripts_fail_closed_without_accessing_user_data(
    script_name,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    script = _load_script(script_name)
    monkeypatch.setattr(
        script,
        "AppSettings",
        lambda: pytest.fail("deprecated script accessed application settings"),
        raising=False,
    )
    monkeypatch.setattr("sys.argv", [script_name])

    with pytest.raises(SystemExit) as stopped:
        script.main()

    message = str(stopped.value)
    assert stopped.value.code not in {None, 0}
    assert "POST /api/videos/{video_id}/reindex" in message
    assert "immutable text-vector generation" in message


def test_visual_index_script_holds_runtime_lock_around_all_shared_state(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("index-visual.py")
    events: list[str] = []

    class FakeSettings:
        data_dir = tmp_path
        database_path = tmp_path / "videoscope.db"
        thumbnails_dir = tmp_path / "thumbnails"
        visual_index_step = 1.0
        siglip_model = "test/siglip"

        @staticmethod
        def ensure_directories() -> None:
            events.append("settings.ensure")

    class FakeLock:
        def __init__(self, data_dir: Path) -> None:
            assert data_dir == tmp_path
            events.append("lock.init")

        def acquire(self) -> None:
            events.append("lock.acquire")

        def close(self) -> None:
            events.append("lock.close")

    class FakeRepository:
        def __init__(self, database_path: Path) -> None:
            assert database_path == FakeSettings.database_path
            events.append("repository.init")

        def initialize(self) -> None:
            assert "lock.acquire" in events
            assert "lock.close" not in events
            events.append("repository.initialize")

        @staticmethod
        def list_videos() -> list[object]:
            return []

    class FakeIndex:
        @staticmethod
        def status(*, check_index: bool):
            assert check_index is False
            events.append("index.status")
            return SimpleNamespace(state=ProviderState.READY, detail="ready")

        @staticmethod
        def close() -> None:
            assert "lock.close" not in events
            events.append("index.close")

    monkeypatch.setattr(script, "AppSettings", FakeSettings)
    monkeypatch.setattr(script, "ExclusiveRuntimeLock", FakeLock, raising=False)
    monkeypatch.setattr(script, "Repository", FakeRepository)
    monkeypatch.setattr(script, "FFmpeg", lambda: SimpleNamespace())
    monkeypatch.setattr(script, "create_visual_index", lambda _settings: FakeIndex())

    script.main()

    assert events.index("lock.acquire") < events.index("repository.initialize")
    assert events[-2:] == ["index.close", "lock.close"]


def test_lighthouse_index_script_holds_runtime_lock_around_all_shared_state(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("index-lighthouse.py")
    events: list[str] = []

    class FakeSettings:
        data_dir = tmp_path
        database_path = tmp_path / "videoscope.db"
        lighthouse_endpoint = None
        lighthouse_allow_in_process = False

        @staticmethod
        def ensure_directories() -> None:
            events.append("settings.ensure")

    class FakeLock:
        def __init__(self, data_dir: Path) -> None:
            assert data_dir == tmp_path
            events.append("lock.init")

        def acquire(self) -> None:
            events.append("lock.acquire")

        def close(self) -> None:
            events.append("lock.close")

    class FakeRepository:
        def __init__(self, database_path: Path) -> None:
            assert database_path == FakeSettings.database_path
            events.append("repository.init")

        def initialize(self) -> None:
            assert "lock.acquire" in events
            assert "lock.close" not in events
            events.append("repository.initialize")

        @staticmethod
        def list_videos() -> list[object]:
            return []

    class FakeRetriever:
        @staticmethod
        def status():  # type: ignore[no-untyped-def]
            events.append("retriever.status")
            return SimpleNamespace(state=ProviderState.READY, detail="ready")

        @staticmethod
        def close() -> None:
            assert "lock.close" not in events
            events.append("retriever.close")

    monkeypatch.setattr(script, "AppSettings", FakeSettings)
    monkeypatch.setattr(script, "ExclusiveRuntimeLock", FakeLock, raising=False)
    monkeypatch.setattr(script, "Repository", FakeRepository)
    monkeypatch.setattr(script, "FFmpeg", lambda: SimpleNamespace())
    monkeypatch.setattr(script, "DisabledLighthouseRetriever", FakeRetriever)

    script.main()

    assert events.index("lock.acquire") < events.index("repository.initialize")
    assert events[-2:] == ["retriever.close", "lock.close"]


@pytest.mark.parametrize("script_name", ["index-visual.py", "index-lighthouse.py"])
def test_index_script_releases_runtime_lock_when_initialization_fails(
    script_name,
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    script = _load_script(script_name)
    events: list[str] = []

    settings = SimpleNamespace(
        data_dir=tmp_path,
        database_path=tmp_path / "videoscope.db",
        ensure_directories=lambda: None,
    )

    class FakeLock:
        def __init__(self, _data_dir: Path) -> None:
            pass

        def acquire(self) -> None:
            events.append("lock.acquire")

        def close(self) -> None:
            events.append("lock.close")

    class FailingRepository:
        def __init__(self, _database_path: Path) -> None:
            pass

        def initialize(self) -> None:
            events.append("repository.initialize")
            raise OSError("database unavailable")

    monkeypatch.setattr(script, "AppSettings", lambda: settings)
    monkeypatch.setattr(script, "ExclusiveRuntimeLock", FakeLock, raising=False)
    monkeypatch.setattr(script, "Repository", FailingRepository)

    with pytest.raises(OSError, match="database unavailable"):
        script.main()

    assert events == ["lock.acquire", "repository.initialize", "lock.close"]


@pytest.mark.parametrize("script_name", ["index-visual.py", "index-lighthouse.py"])
def test_index_script_fails_before_repository_access_when_runtime_is_owned(
    script_name,
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    script = _load_script(script_name)
    settings = SimpleNamespace(
        data_dir=tmp_path,
        database_path=tmp_path / "videoscope.db",
        ensure_directories=lambda: None,
    )
    repository_constructed = False

    class ForbiddenRepository:
        def __init__(self, _database_path: Path) -> None:
            nonlocal repository_constructed
            repository_constructed = True

    monkeypatch.setattr(script, "AppSettings", lambda: settings)
    monkeypatch.setattr(script, "Repository", ForbiddenRepository)
    monkeypatch.setattr(
        script,
        "ExclusiveRuntimeLock",
        ExclusiveRuntimeLock,
        raising=False,
    )

    owner = ExclusiveRuntimeLock(tmp_path)
    owner.acquire()
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            script.main()
    finally:
        owner.close()

    assert repository_constructed is False

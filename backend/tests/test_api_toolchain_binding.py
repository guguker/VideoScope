from types import SimpleNamespace

from fastapi.testclient import TestClient

from videoscope.api import create_app
from videoscope.config import AppSettings
from videoscope.providers.base import ProviderRegistry
from videoscope.repository import Repository
from videoscope.search.service import SearchService
from videoscope.search.vector_index import MemoryVectorIndex


class _Queue:
    def close(self) -> None:
        return None


def test_owned_app_uses_the_runtime_bound_video_index_plan_factory(
    tmp_path,
    monkeypatch,
) -> None:
    settings = AppSettings(_env_file=None, data_dir=tmp_path / "data")
    repository = Repository(settings.database_path)
    repository.initialize()
    expected_plan = object()

    def runtime_plan_factory() -> object:
        return expected_plan

    runtime = SimpleNamespace(
        queue=_Queue(),
        search=SearchService(repository, MemoryVectorIndex()),
        clips=object(),
        providers=ProviderRegistry([]),
        video_index_plan_factory=runtime_plan_factory,
        start=lambda: None,
        close=lambda: True,
    )
    monkeypatch.setattr(
        "videoscope.runtime.build_runtime",
        lambda _settings, _repository: runtime,
    )
    app = create_app(settings=settings, repository=repository)

    with TestClient(app, base_url="http://127.0.0.1"):
        coordinator = app.state.video_index_coordinator
        assert coordinator.plan_factory is runtime_plan_factory

from __future__ import annotations

from dataclasses import replace

import pytest

from videoscope.benchmark.adapter import ProductBenchmarkSearchAdapter
from videoscope.benchmark.catalog import ResolvedAsset
from videoscope.benchmark.profiles import FROZEN_PROFILES
from videoscope.benchmark.runner import BenchmarkSearchHit
from videoscope.benchmark.runner import EXECUTION_LIFECYCLE_COMPONENT_ID
from videoscope.benchmark.schema import ComponentIdentity
from videoscope.search.service import (
    EvaluationSearchConfiguration,
    EvidenceView,
    ProductSearchComponentIdentity,
    ProductSearchIdentities,
    SearchAssetBinding,
    SearchResultView,
)


def _asset(
    asset_id: str = "asset-a",
    video_id: str = "local-video-a",
) -> ResolvedAsset:
    return ResolvedAsset(
        asset_id=asset_id,
        repository_asset_id="sha256:" + "a" * 64,
        video_id=video_id,
        sha256="a" * 64,
        byte_size=100,
        duration_seconds=12.0,
    )


class _RecordingProductSession:
    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0
        self.close_error: BaseException | None = None
        self.lifecycle = "warm:process-cache-preserved@1"
        self.search_calls: list[tuple[str, tuple[str, ...], int]] = []
        self.states: dict[tuple[str, str], str] = {}
        self.results: list[SearchResultView] = []

    def identities(self) -> ProductSearchIdentities:
        return ProductSearchIdentities(
            model=(ProductSearchComponentIdentity("text_embedding", "model@1"),),
            index=(ProductSearchComponentIdentity("text_vectors", "index@1"),),
            config=(ProductSearchComponentIdentity("product_search", "config@1"),),
        )

    def lifecycle_identity(self) -> str:
        return self.lifecycle

    def capability_state(self, video_id: str, capability: str) -> str:
        return self.states.get((video_id, capability), "complete")

    def search(
        self,
        query: str,
        video_ids: tuple[str, ...],
        *,
        limit: int,
    ) -> list[SearchResultView]:
        self.search_calls.append((query, video_ids, limit))
        return self.results

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _FailOnceLifecycleHandle:
    identity = "warm:retryable-lifecycle@1"

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("lifecycle close failed")


class _FailOnceLifecycle:
    def __init__(self) -> None:
        self.handle = _FailOnceLifecycleHandle()

    def begin(self, **_kwargs: object) -> _FailOnceLifecycleHandle:
        return self.handle


class _RecordingSearchService:
    def __init__(self) -> None:
        self.session = _RecordingProductSession()
        self.opened: list[
            tuple[EvaluationSearchConfiguration, tuple[SearchAssetBinding, ...]]
        ] = []

    def open_pinned_evaluation(
        self,
        configuration: EvaluationSearchConfiguration,
        assets: tuple[SearchAssetBinding, ...],
        *,
        execution_mode: str,
        lifecycle_identity: str,
    ) -> _RecordingProductSession:
        assert execution_mode == "warm"
        assert lifecycle_identity.startswith("warm:")
        self.opened.append((configuration, assets))
        return self.session


@pytest.mark.parametrize("profile", FROZEN_PROFILES.values())
def test_adapter_translates_every_frozen_search_plan_without_router_semantics(
    profile,
) -> None:  # type: ignore[no-untyped-def]
    service = _RecordingSearchService()
    adapter = ProductBenchmarkSearchAdapter(service)  # type: ignore[arg-type]

    session = adapter.open_session(profile, (_asset(),), execution_mode="warm")

    configuration, assets = service.opened[0]
    plan = profile.search_plan
    assert configuration.modalities == plan.modalities
    assert configuration.modality_weights == tuple(
        (item.modality, item.weight) for item in plan.modality_weights
    )
    assert configuration.text_search == plan.text_search
    assert configuration.visual_search == plan.visual_search
    assert configuration.temporal_refinement is plan.temporal_refinement
    assert configuration.lighthouse is plan.lighthouse
    assert configuration.reranker == plan.reranker
    assert configuration.reranker_trigger == plan.reranker_trigger
    assert configuration.reranker_candidate_limit == plan.reranker_candidate_limit
    assert configuration.result_limit == plan.result_limit
    assert assets == (
        SearchAssetBinding(
            external_id="asset-a",
            video_id="local-video-a",
            source_sha256="a" * 64,
            byte_size=100,
            duration_seconds=12.0,
        ),
    )
    session.close()


def test_adapter_maps_local_video_ids_back_to_portable_asset_aliases() -> None:
    service = _RecordingSearchService()
    service.session.results = [
        SearchResultView(
            id="private-result-id",
            video_id="local-video-a",
            video_name="private-name.mp4",
            start=2.0,
            end=4.0,
            score=0.75,
            modalities=["speech"],
            evidence=[
                EvidenceView(
                    modality="speech",
                    score=0.75,
                    text="private evidence",
                    source="semantic-generation",
                    confidence=0.9,
                    start=2.0,
                    end=4.0,
                    raw_score=0.7,
                    matched_terms=[],
                    details={},
                )
            ],
            thumbnail_url=None,
            intent="evaluation",
            explanation="frozen evaluation search plan",
            refined=False,
        )
    ]
    adapter = ProductBenchmarkSearchAdapter(service)  # type: ignore[arg-type]
    asset = _asset()
    session = adapter.open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (asset,),
        execution_mode="warm",
    )

    hits = tuple(session.search("spoken phrase", (asset,), limit=20))

    assert hits == (BenchmarkSearchHit("asset-a", 2.0, 4.0, 0.75),)
    assert service.session.search_calls == [
        ("spoken phrase", ("local-video-a",), 20)
    ]


def test_adapter_rejects_results_outside_the_pinned_portable_asset_set() -> None:
    service = _RecordingSearchService()
    service.session.results = [
        SearchResultView(
            id="unexpected",
            video_id="different-local-video",
            video_name="private.mp4",
            start=1.0,
            end=2.0,
            score=0.5,
            modalities=["speech"],
            evidence=[],
            thumbnail_url=None,
            intent="evaluation",
            explanation="frozen evaluation search plan",
            refined=False,
        )
    ]
    adapter = ProductBenchmarkSearchAdapter(service)  # type: ignore[arg-type]
    asset = _asset()
    session = adapter.open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (asset,),
        execution_mode="warm",
    )

    with pytest.raises(RuntimeError, match="unpinned video"):
        tuple(session.search("query", (asset,), limit=20))


def test_adapter_forwards_capabilities_only_for_the_exact_resolved_asset() -> None:
    service = _RecordingSearchService()
    service.session.states[("local-video-a", "text_vectors")] = "stale"
    adapter = ProductBenchmarkSearchAdapter(service)  # type: ignore[arg-type]
    asset = _asset()
    session = adapter.open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (asset,),
        execution_mode="warm",
    )

    assert session.capability_state(asset, "text_vectors") == "stale"
    with pytest.raises(ValueError, match="pinned asset"):
        session.capability_state(
            replace(asset, sha256="b" * 64),
            "text_vectors",
        )


def test_adapter_closes_product_session_and_rejects_duplicate_local_bindings() -> None:
    service = _RecordingSearchService()
    adapter = ProductBenchmarkSearchAdapter(service)  # type: ignore[arg-type]
    session = adapter.open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (_asset(),),
        execution_mode="warm",
    )

    session.close()

    assert service.session.closed
    with pytest.raises(RuntimeError, match="closed"):
        tuple(session.search("query", (_asset(),), limit=20))
    with pytest.raises(ValueError, match="same local video"):
        adapter.open_session(
            FROZEN_PROFILES["lexical_qdrant"],
            (_asset(), _asset("asset-b")),
            execution_mode="warm",
        )


def test_adapter_closes_product_session_when_lifecycle_attestation_mismatches() -> None:
    service = _RecordingSearchService()
    service.session.lifecycle = "warm:different-lifecycle@1"
    adapter = ProductBenchmarkSearchAdapter(service)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="attestation changed"):
        adapter.open_session(
            FROZEN_PROFILES["lexical_qdrant"],
            (_asset(),),
            execution_mode="warm",
        )

    assert service.session.closed


@pytest.mark.parametrize("attestation_failure", ["mismatch", "raises"])
def test_adapter_retains_product_session_when_setup_attestation_cleanup_retries(
    attestation_failure: str,
) -> None:
    class FailThreeCloseProductSession(_RecordingProductSession):
        def lifecycle_identity(self) -> str:
            if attestation_failure == "raises":
                raise RuntimeError("product lifecycle identity failed")
            return "warm:different-lifecycle@1"

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls <= 3:
                raise RuntimeError("product cleanup failed")
            self.closed = True

    class RecordingLifecycleHandle:
        identity = "warm:retained-product-cleanup@1"

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class RecordingLifecycle:
        def __init__(self) -> None:
            self.handle = RecordingLifecycleHandle()

        def begin(self, **_kwargs: object) -> RecordingLifecycleHandle:
            return self.handle

    service = _RecordingSearchService()
    service.session = FailThreeCloseProductSession()
    lifecycle = RecordingLifecycle()
    adapter = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        service,
        lifecycle=lifecycle,
    )

    with pytest.raises(ExceptionGroup) as captured:
        adapter.open_session(
            FROZEN_PROFILES["lexical_qdrant"],
            (_asset(),),
            execution_mode="warm",
        )

    expected_setup_error = (
        "product lifecycle identity failed"
        if attestation_failure == "raises"
        else "product search lifecycle attestation changed"
    )
    assert [str(error) for error in captured.value.exceptions] == [
        expected_setup_error,
        "product cleanup failed",
    ]
    assert service.session.close_calls == 1
    assert lifecycle.handle.close_calls == 0

    adapter.close()
    adapter.close()

    assert service.session.close_calls == 4
    assert service.session.closed
    assert lifecycle.handle.close_calls == 1


def test_default_adapter_rejects_unattested_cold_execution() -> None:
    adapter = ProductBenchmarkSearchAdapter(_RecordingSearchService())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="external attested lifecycle"):
        adapter.open_session(
            FROZEN_PROFILES["lexical_qdrant"],
            (_asset(),),
            execution_mode="cold",
        )


def test_adapter_exposes_converted_identities_and_attested_lifecycle() -> None:
    service = _RecordingSearchService()
    session = ProductBenchmarkSearchAdapter(service).open_session(  # type: ignore[arg-type]
        FROZEN_PROFILES["lexical_qdrant"],
        (_asset(),),
        execution_mode="warm",
    )

    identities = session.identities()

    assert identities.model_identities == (
        ComponentIdentity("text_embedding", "model@1"),
    )
    assert identities.index_identities == (
        ComponentIdentity("text_vectors", "index@1"),
    )
    assert identities.config_identities == (
        ComponentIdentity("product_search", "config@1"),
    )
    assert session.lifecycle_identity() == ComponentIdentity(
        EXECUTION_LIFECYCLE_COMPONENT_ID,
        "warm:process-cache-preserved@1",
    )

    service.session.lifecycle = "warm:drifted@1"
    with pytest.raises(RuntimeError, match="drifted"):
        session.lifecycle_identity()


def test_adapter_persists_product_environment_identity_with_every_session() -> None:
    first_environment = ComponentIdentity(
        "benchmark_product_environment",
        "benchmark-product-environment@1:" + "a" * 64,
    )
    second_environment = ComponentIdentity(
        "benchmark_product_environment",
        "benchmark-product-environment@1:" + "b" * 64,
    )

    first = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        _RecordingSearchService(),
        environment_identity=first_environment,
    ).open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (_asset(),),
        execution_mode="warm",
    )
    second = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        _RecordingSearchService(),
        environment_identity=second_environment,
    ).open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (_asset(),),
        execution_mode="warm",
    )

    assert first.identities().config_identities[-1] == first_environment
    assert second.identities().config_identities[-1] == second_environment
    assert first.identities() != second.identities()

    with pytest.raises(ValueError, match="product environment"):
        ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
            _RecordingSearchService(),
            environment_identity=ComponentIdentity("wrong_component", "value"),
        )


def test_adapter_rejects_invalid_boundary_and_search_contract() -> None:
    with pytest.raises(ValueError, match="pinned evaluation boundary"):
        ProductBenchmarkSearchAdapter(object())  # type: ignore[arg-type]

    service = _RecordingSearchService()
    session = ProductBenchmarkSearchAdapter(service).open_session(  # type: ignore[arg-type]
        FROZEN_PROFILES["lexical_qdrant"],
        (_asset(),),
        execution_mode="warm",
    )

    with pytest.raises(ValueError, match="unique non-empty"):
        tuple(session.search("query", (), limit=20))
    with pytest.raises(ValueError, match="frozen profile"):
        tuple(session.search("query", (_asset(),), limit=19))


def test_adapter_close_retries_lifecycle_without_reclosing_product() -> None:
    service = _RecordingSearchService()
    lifecycle = _FailOnceLifecycle()
    service.session.lifecycle = lifecycle.handle.identity
    session = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        service,
        lifecycle=lifecycle,
    ).open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (_asset(),),
        execution_mode="warm",
    )

    with pytest.raises(RuntimeError, match="lifecycle close failed"):
        session.close()
    with pytest.raises(RuntimeError, match="closing"):
        session.identities()

    session.close()
    session.close()

    assert service.session.close_calls == 1
    assert lifecycle.handle.close_calls == 2


def test_adapter_close_preserves_product_error_across_lifecycle_retry() -> None:
    service = _RecordingSearchService()
    lifecycle = _FailOnceLifecycle()
    service.session.lifecycle = lifecycle.handle.identity
    product_error = RuntimeError("product final validation failed")
    service.session.close_error = product_error
    session = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        service,
        lifecycle=lifecycle,
    ).open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (_asset(),),
        execution_mode="warm",
    )

    with pytest.raises(RuntimeError, match="product final validation failed") as first:
        session.close()
    assert first.value is product_error
    service.session.close_error = None

    with pytest.raises(RuntimeError, match="lifecycle close failed") as second:
        session.close()
    assert second.value.__cause__ is product_error

    with pytest.raises(RuntimeError, match="product final validation failed") as third:
        session.close()
    assert third.value is product_error
    session.close()

    assert service.session.close_calls == 2
    assert lifecycle.handle.close_calls == 2


def test_adapter_retains_and_drains_a_successful_session_after_close_failures() -> None:
    class FailTwiceLifecycleHandle:
        identity = "warm:retained-successful-session@1"

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls <= 2:
                raise RuntimeError("lifecycle close failed")

    class FailTwiceLifecycle:
        def __init__(self) -> None:
            self.handle = FailTwiceLifecycleHandle()

        def begin(self, **_kwargs: object) -> FailTwiceLifecycleHandle:
            return self.handle

    service = _RecordingSearchService()
    lifecycle = FailTwiceLifecycle()
    service.session.lifecycle = lifecycle.handle.identity
    adapter = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        service,
        lifecycle=lifecycle,
    )
    session = adapter.open_session(
        FROZEN_PROFILES["lexical_qdrant"],
        (_asset(),),
        execution_mode="warm",
    )

    with pytest.raises(RuntimeError, match="lifecycle close failed"):
        session.close()
    adapter.close()
    adapter.close()

    assert service.session.close_calls == 1
    assert lifecycle.handle.close_calls == 3


def test_adapter_retries_lifecycle_cleanup_when_product_session_open_fails() -> None:
    lifecycle = _FailOnceLifecycle()

    class FailingSearchService(_RecordingSearchService):
        def open_pinned_evaluation(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            raise RuntimeError("product session open failed")

    adapter = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        FailingSearchService(),
        lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError, match="product session open failed"):
        adapter.open_session(
            FROZEN_PROFILES["lexical_qdrant"],
            (_asset(),),
            execution_mode="warm",
        )

    assert lifecycle.handle.close_calls == 2


def test_adapter_closes_lifecycle_when_reading_its_identity_fails() -> None:
    class RaisingIdentityHandle:
        def __init__(self) -> None:
            self.close_calls = 0

        @property
        def identity(self) -> str:
            raise RuntimeError("lifecycle identity failed")

        def close(self) -> None:
            self.close_calls += 1

    class RaisingIdentityLifecycle:
        def __init__(self) -> None:
            self.handle = RaisingIdentityHandle()

        def begin(self, **_kwargs: object) -> RaisingIdentityHandle:
            return self.handle

    lifecycle = RaisingIdentityLifecycle()
    adapter = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        _RecordingSearchService(),
        lifecycle=lifecycle,
    )

    with pytest.raises(RuntimeError, match="lifecycle identity failed"):
        adapter.open_session(
            FROZEN_PROFILES["lexical_qdrant"],
            (_asset(),),
            execution_mode="warm",
        )

    assert lifecycle.handle.close_calls == 1


def test_adapter_retains_failed_setup_cleanup_for_explicit_retry() -> None:
    class InitiallyFailingHandle:
        identity = "warm:persistent-cleanup@1"

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls <= 3:
                raise RuntimeError("persistent lifecycle cleanup failure")

    class InitiallyFailingLifecycle:
        def __init__(self) -> None:
            self.handle = InitiallyFailingHandle()

        def begin(self, **_kwargs: object) -> InitiallyFailingHandle:
            return self.handle

    class FailingSearchService(_RecordingSearchService):
        def open_pinned_evaluation(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            raise RuntimeError("product session open failed")

    lifecycle = InitiallyFailingLifecycle()
    adapter = ProductBenchmarkSearchAdapter(  # type: ignore[arg-type]
        FailingSearchService(),
        lifecycle=lifecycle,
    )

    with pytest.raises(ExceptionGroup) as captured:
        adapter.open_session(
            FROZEN_PROFILES["lexical_qdrant"],
            (_asset(),),
            execution_mode="warm",
        )

    assert lifecycle.handle.close_calls == 3
    assert [str(error) for error in captured.value.exceptions] == [
        "product session open failed",
        "benchmark lifecycle cleanup failed after session setup",
    ]

    adapter.close()
    adapter.close()

    assert lifecycle.handle.close_calls == 4

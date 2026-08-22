from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from videoscope.search.service import (
    EvaluationSearchConfiguration,
    PinnedProductSearchSession,
    ProductSearchIdentities,
    SearchAssetBinding,
    SearchResultView,
    SearchService,
)

from .catalog import ResolvedAsset
from .profiles import BenchmarkProfile
from .runner import (
    EXECUTION_LIFECYCLE_COMPONENT_ID,
    BenchmarkSearchHit,
    ExecutionIdentities,
)
from .schema import ComponentIdentity


_LIFECYCLE_SETUP_CLEANUP_ATTEMPTS = 3
_RETAINED_SESSION_CLEANUP_ATTEMPTS = 4
BENCHMARK_PRODUCT_ENVIRONMENT_COMPONENT_ID = "benchmark_product_environment"


class ProductBenchmarkLifecycleHandle(Protocol):
    @property
    def identity(self) -> str: ...

    def close(self) -> None: ...


class ProductBenchmarkLifecycle(Protocol):
    def begin(
        self,
        *,
        execution_mode: Literal["cold", "warm"],
        profile: BenchmarkProfile,
        assets: tuple[ResolvedAsset, ...],
    ) -> ProductBenchmarkLifecycleHandle: ...


@dataclass(slots=True)
class _WarmLifecycleHandle:
    identity: str = "warm:process-cache-preserved@1"

    def close(self) -> None:
        return None


class WarmProcessCacheLifecycle:
    """Honest default: preserve process caches and reject unsupported cold runs."""

    def begin(
        self,
        *,
        execution_mode: Literal["cold", "warm"],
        profile: BenchmarkProfile,
        assets: tuple[ResolvedAsset, ...],
    ) -> ProductBenchmarkLifecycleHandle:
        del profile, assets
        if execution_mode != "warm":
            raise RuntimeError(
                "cold benchmark execution requires an external attested lifecycle"
            )
        return _WarmLifecycleHandle()


class ProductBenchmarkSearchAdapter:
    """Translate frozen benchmark plans into generation-pinned product search."""

    def __init__(
        self,
        search: SearchService,
        *,
        lifecycle: ProductBenchmarkLifecycle | None = None,
        environment_identity: ComponentIdentity | None = None,
    ) -> None:
        open_pinned = getattr(search, "open_pinned_evaluation", None)
        if not callable(open_pinned):
            raise ValueError("product search does not expose a pinned evaluation boundary")
        if environment_identity is not None and (
            not isinstance(environment_identity, ComponentIdentity)
            or environment_identity.component_id
            != BENCHMARK_PRODUCT_ENVIRONMENT_COMPONENT_ID
        ):
            raise ValueError("benchmark product environment identity is invalid")
        self._search = search
        self._lifecycle = lifecycle or WarmProcessCacheLifecycle()
        self._environment_identity = environment_identity
        self._failed_lifecycle_handles: list[ProductBenchmarkLifecycleHandle] = []
        self._sessions: list[ProductBenchmarkSearchSession] = []

    @staticmethod
    def _configuration(profile: BenchmarkProfile) -> EvaluationSearchConfiguration:
        if not isinstance(profile, BenchmarkProfile):
            raise ValueError("benchmark profile must be validated")
        plan = profile.search_plan
        configuration = EvaluationSearchConfiguration(
            modalities=plan.modalities,
            modality_weights=tuple(
                (item.modality, item.weight) for item in plan.modality_weights
            ),
            text_search=plan.text_search,
            visual_search=plan.visual_search,
            temporal_refinement=plan.temporal_refinement,
            lighthouse=plan.lighthouse,
            reranker=plan.reranker,
            reranker_trigger=plan.reranker_trigger,
            reranker_candidate_limit=plan.reranker_candidate_limit,
            result_limit=plan.result_limit,
            schema_version=plan.schema_version,
        )
        if configuration.canonical_json != plan.canonical_json:
            raise ValueError("benchmark search plan translation is not exact")
        return configuration

    @staticmethod
    def _asset_bindings(
        assets: tuple[ResolvedAsset, ...],
    ) -> tuple[SearchAssetBinding, ...]:
        if (
            type(assets) is not tuple
            or not assets
            or any(not isinstance(asset, ResolvedAsset) for asset in assets)
        ):
            raise ValueError("benchmark assets must be validated")
        if len({asset.asset_id for asset in assets}) != len(assets):
            raise ValueError("benchmark asset aliases must be unique")
        if len({asset.video_id for asset in assets}) != len(assets):
            raise ValueError("benchmark aliases cannot bind the same local video")
        return tuple(
            SearchAssetBinding(
                external_id=asset.asset_id,
                video_id=asset.video_id,
                source_sha256=asset.sha256,
                byte_size=asset.byte_size,
                duration_seconds=asset.duration_seconds,
            )
            for asset in assets
        )

    def open_session(
        self,
        profile: BenchmarkProfile,
        assets: tuple[ResolvedAsset, ...],
        *,
        execution_mode: Literal["cold", "warm"],
    ) -> ProductBenchmarkSearchSession:
        if execution_mode not in {"cold", "warm"}:
            raise ValueError("unsupported benchmark execution mode")
        self.close()
        configuration = self._configuration(profile)
        bindings = self._asset_bindings(assets)
        lifecycle: ProductBenchmarkLifecycleHandle | None = None
        session: ProductBenchmarkSearchSession | None = None
        try:
            lifecycle = self._lifecycle.begin(
                execution_mode=execution_mode,
                profile=profile,
                assets=assets,
            )
            lifecycle_identity = lifecycle.identity
            if (
                type(lifecycle_identity) is not str
                or not lifecycle_identity.startswith(f"{execution_mode}:")
            ):
                raise ValueError(
                    "benchmark lifecycle identity does not attest its mode"
                )
            product_session = self._search.open_pinned_evaluation(
                configuration,
                bindings,
                execution_mode=execution_mode,
                lifecycle_identity=lifecycle_identity,
            )
            session = ProductBenchmarkSearchSession(
                profile=profile,
                assets=assets,
                product_session=product_session,
                lifecycle=lifecycle,
                environment_identity=self._environment_identity,
            )
            self._sessions.append(session)
            if product_session.lifecycle_identity() != lifecycle_identity:
                raise ValueError("product search lifecycle attestation changed")
        except BaseException as setup_error:
            if session is not None:
                try:
                    session.close()
                except BaseException as cleanup_error:
                    self._raise_setup_cleanup_failure(
                        setup_error,
                        cleanup_error,
                        message="benchmark session setup and product cleanup failed",
                    )
                self._sessions = [
                    retained for retained in self._sessions if retained is not session
                ]
                raise
            if lifecycle is not None:
                try:
                    self._close_lifecycle_after_setup_failure(lifecycle)
                except BaseException as cleanup_error:
                    self._failed_lifecycle_handles.append(lifecycle)
                    self._raise_setup_cleanup_failure(
                        setup_error,
                        cleanup_error,
                        message=(
                            "benchmark session setup and lifecycle cleanup failed"
                        ),
                    )
            raise
        assert session is not None
        return session

    def close(self) -> None:
        failures: list[BaseException] = []
        for session in tuple(self._sessions):
            last_error: BaseException | None = None
            for _attempt in range(_RETAINED_SESSION_CLEANUP_ATTEMPTS):
                try:
                    session.close()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as error:
                    last_error = error
                    continue
                self._sessions = [
                    retained for retained in self._sessions if retained is not session
                ]
                break
            else:
                assert last_error is not None
                failures.append(last_error)
        for lifecycle in tuple(self._failed_lifecycle_handles):
            try:
                self._close_lifecycle_after_setup_failure(lifecycle)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                failures.append(error)
                continue
            self._failed_lifecycle_handles = [
                retained
                for retained in self._failed_lifecycle_handles
                if retained is not lifecycle
            ]
        if failures:
            raise RuntimeError(
                "benchmark adapter retained lifecycle cleanup failed"
            ) from failures[0]

    @staticmethod
    def _raise_setup_cleanup_failure(
        setup_error: BaseException,
        cleanup_error: BaseException,
        *,
        message: str,
    ) -> None:
        if isinstance(setup_error, (KeyboardInterrupt, SystemExit)):
            raise setup_error from cleanup_error
        if isinstance(cleanup_error, (KeyboardInterrupt, SystemExit)):
            raise cleanup_error from setup_error
        if isinstance(setup_error, Exception) and isinstance(
            cleanup_error,
            Exception,
        ):
            raise ExceptionGroup(
                message,
                [setup_error, cleanup_error],
            ) from setup_error
        raise cleanup_error from setup_error

    @staticmethod
    def _close_lifecycle_after_setup_failure(
        lifecycle: ProductBenchmarkLifecycleHandle,
    ) -> None:
        close = getattr(lifecycle, "close", None)
        if not callable(close):
            raise RuntimeError("benchmark lifecycle cleanup contract is invalid")
        last_error: BaseException | None = None
        for _attempt in range(_LIFECYCLE_SETUP_CLEANUP_ATTEMPTS):
            try:
                close()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as error:
                last_error = error
                continue
            return
        assert last_error is not None
        raise RuntimeError(
            "benchmark lifecycle cleanup failed after session setup"
        ) from last_error


class ProductBenchmarkSearchSession:
    def __init__(
        self,
        *,
        profile: BenchmarkProfile,
        assets: tuple[ResolvedAsset, ...],
        product_session: PinnedProductSearchSession,
        lifecycle: ProductBenchmarkLifecycleHandle,
        environment_identity: ComponentIdentity | None,
    ) -> None:
        self._profile = profile
        self._assets = tuple(assets)
        self._assets_by_alias = {asset.asset_id: asset for asset in assets}
        self._aliases_by_video = {asset.video_id: asset.asset_id for asset in assets}
        self._product_session = product_session
        self._lifecycle = lifecycle
        self._environment_identity = environment_identity
        self._closed = False
        self._product_closed = False
        self._lifecycle_closed = False
        self._product_close_error: BaseException | None = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("benchmark product search session is closed")
        if self._product_closed or self._product_close_error is not None:
            raise RuntimeError("benchmark product search session is closing")

    def _require_asset(self, asset: ResolvedAsset) -> ResolvedAsset:
        if not isinstance(asset, ResolvedAsset):
            raise ValueError("benchmark search uses an invalid asset")
        expected = self._assets_by_alias.get(asset.asset_id)
        if expected is None or asset != expected:
            raise ValueError("benchmark search asset differs from the pinned asset")
        return expected

    def identities(self) -> ExecutionIdentities:
        self._ensure_open()
        product: ProductSearchIdentities = self._product_session.identities()

        def convert(values):  # type: ignore[no-untyped-def]
            return tuple(
                ComponentIdentity(item.component_id, item.identity) for item in values
            )

        config_identities = convert(product.config)
        if self._environment_identity is not None:
            if any(
                identity.component_id
                == BENCHMARK_PRODUCT_ENVIRONMENT_COMPONENT_ID
                for identity in config_identities
            ):
                raise RuntimeError(
                    "product search duplicated the benchmark environment identity"
                )
            config_identities = (*config_identities, self._environment_identity)
        return ExecutionIdentities(
            model_identities=convert(product.model),
            index_identities=convert(product.index),
            config_identities=config_identities,
        )

    def lifecycle_identity(self) -> ComponentIdentity:
        self._ensure_open()
        identity = self._product_session.lifecycle_identity()
        if identity != self._lifecycle.identity:
            raise RuntimeError("benchmark lifecycle attestation drifted")
        return ComponentIdentity(EXECUTION_LIFECYCLE_COMPONENT_ID, identity)

    def capability_state(
        self,
        asset: ResolvedAsset,
        capability: str,
    ) -> str:
        self._ensure_open()
        pinned = self._require_asset(asset)
        return self._product_session.capability_state(
            pinned.video_id,
            capability,
        )

    def search(
        self,
        query: str,
        assets: tuple[ResolvedAsset, ...],
        *,
        limit: int,
    ) -> tuple[BenchmarkSearchHit, ...]:
        self._ensure_open()
        if (
            type(assets) is not tuple
            or not assets
            or len({asset.asset_id for asset in assets}) != len(assets)
        ):
            raise ValueError("benchmark search assets must be a unique non-empty tuple")
        pinned_assets = tuple(self._require_asset(asset) for asset in assets)
        if limit != self._profile.search_plan.result_limit:
            raise ValueError("benchmark search limit differs from the frozen profile")
        results: list[SearchResultView] = self._product_session.search(
            query,
            tuple(asset.video_id for asset in pinned_assets),
            limit=limit,
        )
        if len(results) > limit:
            raise RuntimeError("product search exceeded the frozen result limit")
        hits: list[BenchmarkSearchHit] = []
        selected_aliases = {asset.asset_id for asset in pinned_assets}
        for result in results:
            alias = self._aliases_by_video.get(result.video_id)
            if alias is None or alias not in selected_aliases:
                raise RuntimeError("product search returned an unpinned video")
            hits.append(
                BenchmarkSearchHit(
                    asset_id=alias,
                    start_seconds=result.start,
                    end_seconds=result.end,
                    score=result.score,
                )
            )
        return tuple(hits)

    def close(self) -> None:
        if self._closed:
            return
        if not self._product_closed:
            try:
                self._product_session.close()
            except BaseException as error:
                if self._product_close_error is None:
                    self._product_close_error = error
                raise
            self._product_closed = True
        if not self._lifecycle_closed:
            try:
                self._lifecycle.close()
            except BaseException as lifecycle_error:
                if self._product_close_error is not None:
                    raise lifecycle_error from self._product_close_error
                raise
            self._lifecycle_closed = True
        self._closed = True
        if self._product_close_error is not None:
            raise self._product_close_error

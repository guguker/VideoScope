from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

import videoscope.benchmark.phase0_evidence as phase0_evidence_module
from videoscope.benchmark.phase0_evidence import (
    REQUIRED_EXECUTED_PROFILE_IDS,
    Phase0EvidenceError,
    ROLLBACK_FORCED_FAILURE,
    ROLLBACK_PROOF_ID,
    ROLLBACK_SEMANTIC_ASSERTIONS,
    ROLLBACK_TEST_NODE_ID,
    ROLLBACK_TEST_SOURCE,
    build_phase0_evidence,
    collect_phase0_evidence,
    validate_baseline_batch,
    validate_benchmark_runs,
    validate_full_ml_smoke,
    validate_ml_environment_attestation,
    validate_rollback_proof,
)
from videoscope.benchmark.measurements import measurement_metrics_from_evidence
from videoscope.benchmark.metric_policy import load_frozen_metric_policy
from videoscope.benchmark.profiles import get_profile, profile_identity_contract
from videoscope.benchmark.runner import (
    EXECUTION_LIFECYCLE_COMPONENT_ID,
    BenchmarkRunner,
    BenchmarkSearchHit,
    ExecutionIdentities,
    _methodology_identity,
    _score_ranked_hits,
)
from videoscope.benchmark.schema import (
    MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
    MEASUREMENT_PROTOCOL_COMPONENT_ID,
    RUN_SCHEMA_VERSION,
    BenchmarkMeasurementEvidence,
    BenchmarkRunManifest,
    BenchmarkStorageSnapshot,
    ComponentIdentity,
    HardwareProfile,
)
from videoscope.benchmark.serialization import dataset_revision, run_to_dict
from videoscope.benchmark.storage import load_dataset
from videoscope.benchmark.video_verifier_runner import (
    VIDEO_VERIFIER_RUN_SCHEMA_VERSION,
    VideoVerifierAttempt,
    VideoVerifierPrediction,
    VideoVerifierRunManifest,
    VideoVerifierSummary,
    _run_to_dict as video_verifier_run_to_dict,
)
from videoscope.benchmark.video_verifier_schema import (
    video_verifier_dataset_from_json,
    video_verifier_dataset_revision,
)
from videoscope.evaluation import temporal_iou


_CODE_SHA = "a" * 40
_DIGEST = "b" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "docs/benchmarks/policies/phase0-regression-v1.json"
PRODUCT_DATASET_PATH = (
    PROJECT_ROOT / "docs/benchmarks/product-retrieval/seed-v1.json"
)
VERIFIER_DATASET_PATH = PROJECT_ROOT / "docs/benchmarks/video-verifier/seed-v1.json"

_TEXT_MODEL_IDENTITY = (
    "fastembed@0.8.0:mean-pooling-v1:"
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2:"
    "xenova/paraphrase-multilingual-mpnet-base-v2@"
    "e5d116277351513fd260955ece953ecddde7046e:768"
)
_VISUAL_MODEL_IDENTITY = (
    "google/siglip2-base-patch16-224@"
    "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
)
_QWEN_MODEL_IDENTITY = (
    "mlx-community/Qwen3.5-9B-MLX-4bit@"
    "938d8919941c6e7efd3c7150eff7fe9d12afa631"
)
_COMPONENT_EXECUTION_IDS = (
    "text_vectors",
    "lexical_text",
    "visual_dense",
    "temporal_refinement",
    "lighthouse",
    "qwen_verification",
)


def _selected_component_ids(profile_id: str) -> tuple[str, ...]:
    plan = get_profile(profile_id).search_plan
    selected: list[str] = []
    if plan.text_search != "disabled":
        selected.extend(("text_vectors", "lexical_text"))
    if plan.visual_search != "disabled":
        selected.append("visual_dense")
    if plan.temporal_refinement:
        selected.append("temporal_refinement")
    if plan.lighthouse:
        selected.append("lighthouse")
    if plan.reranker == "qwen":
        selected.append("qwen_verification")
    return tuple(selected)


def _component_execution(profile_id: str) -> dict[str, object]:
    profile = get_profile(profile_id)
    selected = set(_selected_component_ids(profile_id))
    counts = {
        component_id: int(component_id in selected)
        for component_id in _COMPONENT_EXECUTION_IDS
    }
    return {
        "schema_version": 1,
        "profile_identity": profile.identity,
        "search_configuration_identity": profile.search_plan.identity.replace(
            "evaluation-search-plan",
            "evaluation-search-configuration",
            1,
        ),
        "invoked_component_ids": list(_selected_component_ids(profile_id)),
        "component_input_counts": dict(counts),
        "component_output_counts": dict(counts),
        "component_evidence_counts": dict(counts),
    }


def _environment_report() -> dict[str, object]:
    expected = phase0_evidence_module._current_ml_environment_contract()
    expected_host = expected["host"]
    return {
        "attestation_id": expected["attestation_id"],
        "capabilities": deepcopy(expected["capabilities"]),
        "contracts": deepcopy(expected["contracts"]),
        "environments": deepcopy(expected["environments"]),
        "failures": [],
        "host": {
            "chip": expected_host["chip"],  # type: ignore[index]
            "machine": expected_host["machine"],  # type: ignore[index]
            "macos_major": 26,
            "memory_gib": expected_host["memory_gib"],  # type: ignore[index]
            "status": "complete",
            "system": expected_host["system"],  # type: ignore[index]
        },
        "manifest_identity": expected["manifest_identity"],
        "offline_environment": {
            "flags": list(expected["offline_flags"]),  # type: ignore[arg-type]
            "status": "complete",
        },
        "schema_version": 1,
        "status": "complete",
        "uv": deepcopy(expected["uv"]),
    }


def _host_resources() -> dict[str, object]:
    return {
        "schema_version": 1,
        "provider_identity": (
            "darwin-iokit-ioaccelerator-performance-statistics-"
            "mach-vm-statistics64@1"
        ),
        "scope": "system_wide",
        "memory_accounting": "metal_standalone_not_additive_with_process_rss",
        "sample_interval_milliseconds": 250,
        "raw_samples": [
            {
                "elapsed_nanoseconds": 0,
                "metal": {
                    "in_use_system_memory_bytes": 100,
                    "alloc_system_memory_bytes": 200,
                    "recovery_count": 4,
                },
                "virtual_memory": {
                    "swapins_pages": 10,
                    "swapouts_pages": 20,
                    "page_size_bytes": 16_384,
                },
            },
            {
                "elapsed_nanoseconds": 250_000_000,
                "metal": {
                    "in_use_system_memory_bytes": 140,
                    "alloc_system_memory_bytes": 250,
                    "recovery_count": 4,
                },
                "virtual_memory": {
                    "swapins_pages": 10,
                    "swapouts_pages": 20,
                    "page_size_bytes": 16_384,
                },
            },
        ],
        "metal": {
            "in_use_system_memory": {
                "baseline_bytes": 100,
                "peak_bytes": 140,
                "increment_bytes": 40,
            },
            "alloc_system_memory": {
                "baseline_bytes": 200,
                "peak_bytes": 250,
                "increment_bytes": 50,
            },
            "recovery_delta": 0,
        },
        "virtual_memory": {
            "swapins_delta_pages": 0,
            "swapins_delta_bytes": 0,
            "swapouts_delta_pages": 0,
            "swapouts_delta_bytes": 0,
            "page_size_bytes": 16_384,
        },
    }


def _full_ml_smoke() -> dict[str, object]:
    environment = phase0_evidence_module._current_ml_environment_contract()
    executed = {
        profile: {
            "close": "complete",
            "component_execution": _component_execution(profile),
            "evidence_count": 1,
            "generation_bound": True,
            "open": "complete",
            "search": "complete",
            "status": "complete",
        }
        for profile in (
            "lexical_qdrant",
            "dense_siglip",
            "temporal_refinement",
            "lighthouse",
            "qwen_verification",
        )
    }
    executed["internvideo"] = {
        "reason_code": "provider_not_configured",
        "status": "not_configured",
    }
    host = _host_resources()
    return {
        "code_sha_after": _CODE_SHA,
        "code_sha_before": _CODE_SHA,
        "environment_bindings": dict(
            phase0_evidence_module.FULL_ML_SMOKE_ENVIRONMENT_BINDINGS
        ),
        "ml_environment_attestation_id": environment["attestation_id"],
        "ml_environment_manifest_identity": environment["manifest_identity"],
        "offline": True,
        "product_integration": {
            "status": "complete",
            "upload": "complete",
            "index": "complete",
            "search": "complete",
            "export": {
                "byte_size": 1000,
                "container": "mp4",
                "duration_seconds": 1.0,
                "source_profile_id": "lexical_qdrant",
                "status": "complete",
            },
            "generation_bound": True,
            "evidence_count": 5,
            "profiles": executed,
            "runtime_cleanup": "complete",
        },
        "resources": {
            "host_resources": host,
            "measurement_caveat": {
                "memory_accounting": (
                    "metal_standalone_not_additive_with_process_rss"
                ),
                "scope": "system_wide",
            },
            "oom": {"status": "not_observed"},
            "peak_metal_bytes": {
                "baseline_bytes": 100,
                "increment_bytes": 40,
                "scope": "system_wide",
                "status": "measured",
                "value": 140,
            },
            "sampled_peak_process_tree_rss_bytes": {
                "baseline_bytes": 1_000,
                "external_loopback_workers_included": True,
                "increment_bytes": 500,
                "managed_worker_roles": [
                    "vision",
                    "whisper",
                    "lighthouse",
                    "qwen",
                ],
                "sample_count": 3,
                "samples_bytes": [1_000, 1_500, 1_200],
                "sampling_interval_ms": 50,
                "scope": "smoke_process_and_descendants",
                "status": "measured",
                "value": 1_500,
            },
            "system_wide_pressure_deltas": {
                "metal_recovery_count": 0,
                "swapins_bytes": 0,
                "swapins_pages": 0,
                "swapouts_bytes": 0,
                "swapouts_pages": 0,
            },
        },
        "schema_version": 2,
        "status": "ready",
        "steps": [
            {
                "id": "vision.image_embedding",
                "observations": {"dimensions": 768, "vector_count": 1},
                "status": "complete",
            },
            {
                "id": "vision.text_embedding",
                "observations": {"dimensions": 768, "vector_count": 1},
                "status": "complete",
            },
            {
                "id": "vision.rfdetr",
                "observations": {"detection_count": 0},
                "status": "complete",
            },
            {
                "id": "whisper.transcribe",
                "observations": {"segment_count": 0},
                "status": "complete",
            },
            {
                "id": "ocr.read",
                "observations": {"item_count": 0},
                "status": "complete",
            },
            {
                "id": "lighthouse.generation",
                "observations": {"generation_count": 1},
                "status": "complete",
            },
            {
                "id": "lighthouse.search",
                "observations": {"hit_count": 1},
                "status": "complete",
            },
            {
                "id": "qwen.judge",
                "observations": {"judgement_count": 1},
                "status": "complete",
            },
        ],
        "toolchain_identity": "sha256:" + _DIGEST,
        "workspace_cleanup": "complete",
    }


def _rollback_proof() -> dict[str, object]:
    source = Path(__file__).resolve().parents[2] / ROLLBACK_TEST_SOURCE
    return {
        "schema_version": 1,
        "proof_id": ROLLBACK_PROOF_ID,
        "code_sha": _CODE_SHA,
        "forced_failure": ROLLBACK_FORCED_FAILURE,
        "semantic_assertions": list(ROLLBACK_SEMANTIC_ASSERTIONS),
        "status": "verified",
        "test_node_id": ROLLBACK_TEST_NODE_ID,
        "test_result": "passed",
        "test_source_sha256": sha256(source.read_bytes()).hexdigest(),
    }


def _measurement_evidence() -> BenchmarkMeasurementEvidence:
    active = BenchmarkStorageSnapshot(
        root_id="active-artifacts",
        purpose="active_immutable_artifacts",
        file_count=3,
        directory_count=2,
        logical_bytes=1_024,
        allocated_bytes=4_096,
        tree_digest="1" * 64,
    )
    scratch_before = BenchmarkStorageSnapshot(
        root_id="benchmark-scratch",
        purpose="benchmark_scratch",
        file_count=0,
        directory_count=1,
        logical_bytes=0,
        allocated_bytes=0,
        tree_digest="2" * 64,
    )
    scratch_after = BenchmarkStorageSnapshot(
        root_id="benchmark-scratch",
        purpose="benchmark_scratch",
        file_count=1,
        directory_count=1,
        logical_bytes=512,
        allocated_bytes=4_096,
        tree_digest="3" * 64,
    )
    return BenchmarkMeasurementEvidence(
        schema_version=MEASUREMENT_EVIDENCE_SCHEMA_VERSION,
        rss_samples_bytes=(100_000_000, 120_000_000, 110_000_000),
        storage_before=(active, scratch_before),
        storage_after=(active, scratch_after),
        metal_telemetry_status="unavailable",
    )


def _benchmark_identities(profile_id: str) -> ExecutionIdentities:
    profile = get_profile(profile_id)
    models = [ComponentIdentity("text_embedding", _TEXT_MODEL_IDENTITY)]
    indexes = [
        ComponentIdentity("text_vector_index", "1" * 64),
        ComponentIdentity("text_vector_generations", "sha256:" + "2" * 64),
    ]
    if profile.search_plan.visual_search != "disabled":
        models.append(ComponentIdentity("visual_embedding", _VISUAL_MODEL_IDENTITY))
        indexes.append(
            ComponentIdentity("visual_generations", "sha256:" + "4" * 64)
        )
    if profile.search_plan.lighthouse:
        models.append(ComponentIdentity("lighthouse_model", "sha256:" + "5" * 64))
        indexes.append(
            ComponentIdentity("lighthouse_generations", "sha256:" + "6" * 64)
        )
    if profile.search_plan.reranker != "none":
        models.append(
            ComponentIdentity(
                f"{profile.search_plan.reranker}_reranker",
                _QWEN_MODEL_IDENTITY,
            )
        )
    environment_digest = sha256(
        f"environment:{profile.identity}".encode("utf-8")
    ).hexdigest()
    runtime_digest = sha256(f"runtime:{profile.identity}".encode("utf-8")).hexdigest()
    lifecycle = "warm:test-cache-policy@1"
    configs = (
        ComponentIdentity(
            "benchmark_product_environment",
            f"benchmark-product-environment@2:{environment_digest}",
        ),
        ComponentIdentity(
            "evaluation_search_configuration",
            profile.search_plan.identity.replace(
                "evaluation-search-plan",
                "evaluation-search-configuration",
                1,
            ),
        ),
        ComponentIdentity("product_search_lifecycle", lifecycle),
        ComponentIdentity("product_search_runtime", "sha256:" + runtime_digest),
        ComponentIdentity(EXECUTION_LIFECYCLE_COMPONENT_ID, lifecycle),
        ComponentIdentity("benchmark_profile", profile.identity),
        ComponentIdentity("benchmark_search_plan", profile.search_plan.identity),
        ComponentIdentity("benchmark_methodology", _methodology_identity()),
        ComponentIdentity(
            "benchmark_profile_identity_contract",
            profile_identity_contract(profile).identity,
        ),
    )
    return ExecutionIdentities(tuple(models), tuple(indexes), configs)


def _benchmark_run(profile_id: str) -> BenchmarkRunManifest:
    dataset = load_dataset(PRODUCT_DATASET_PATH)
    scores = []
    outcomes = []
    for case in sorted(dataset.cases, key=lambda item: item.case_id):
        interval = case.relevant_intervals[0]
        score = _score_ranked_hits(
            case,
            (
                BenchmarkSearchHit(
                    interval.asset_id,
                    interval.start_seconds,
                    interval.end_seconds,
                    1.0,
                ),
            ),
            latency_ms=10.0,
            overlap=temporal_iou,
        )
        scores.append(score)
        outcomes.append(BenchmarkRunner._complete_outcome(score))
    evidence = _measurement_evidence()
    identities = _benchmark_identities(profile_id)
    return BenchmarkRunManifest(
        schema_version=RUN_SCHEMA_VERSION,
        run_id=f"phase0-{profile_id}",
        created_at="2026-09-04T00:00:03Z",
        started_at="2026-09-04T00:00:00Z",
        finished_at="2026-09-04T00:00:02Z",
        run_status="complete",
        code_sha=_CODE_SHA,
        dataset_revision=dataset_revision(dataset),
        model_identities=identities.model_identities,
        index_identities=identities.index_identities,
        config_identities=identities.config_identities,
        hardware=HardwareProfile(
            operating_system="Darwin 26",
            architecture="arm64",
            processor="Apple M4 Pro",
            memory_bytes=24 * 1024**3,
            accelerator="Metal",
        ),
        execution_mode="warm",
        quality_metrics=BenchmarkRunner._run_metrics(
            dataset,
            tuple(outcomes),
            tuple(scores),
        ),
        system_metrics=measurement_metrics_from_evidence(evidence),
        measurement_protocol=ComponentIdentity(
            MEASUREMENT_PROTOCOL_COMPONENT_ID,
            "process-tree-rss-50ms-contained-storage@2:" + "7" * 64,
        ),
        measurement_status="complete",
        measurement_started_at="2026-09-04T00:00:00Z",
        measurement_finished_at="2026-09-04T00:00:01Z",
        case_outcomes=tuple(outcomes),
        measurement_evidence_status="complete",
        measurement_evidence=evidence,
    )


def _benchmark_runs() -> dict[str, BenchmarkRunManifest]:
    return {
        profile_id: _benchmark_run(profile_id)
        for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
    }


def _video_verifier_run() -> VideoVerifierRunManifest:
    dataset = video_verifier_dataset_from_json(VERIFIER_DATASET_PATH.read_bytes())
    environment = phase0_evidence_module._current_ml_environment_contract()
    qwen_capability = next(
        item
        for item in environment["capabilities"]  # type: ignore[union-attr]
        if item["id"] == "qwen_verification"
    )
    attempts = tuple(
        VideoVerifierAttempt(
            case_id=case.case_id,
            candidate_id=f"candidate-{rank}",
            prepared_input_sha256=case.prepared_input_sha256,
            prepared_input_byte_size=case.prepared_input_byte_size,
            proposal_rank=rank,
            proposal_score=0.5,
            status="match",
            latency_ms=20.0,
            prediction=VideoVerifierPrediction(
                facts=case.expected_facts,
                predicted_jersey=case.expected_jersey,
                confidence=0.9,
            ),
            error_code=None,
        )
        for rank, case in enumerate(dataset.cases, start=1)
    )
    identity = json.dumps(
        {
            "contract": phase0_evidence_module.QWEN_WORKER_SCHEMA_VERSION,
            "fps": 2.0,
            "input_root_sha256": "8" * 64,
            "max_tokens": 320,
            "mode": "isolated-worker",
            "model": _QWEN_MODEL_IDENTITY,
            "prompt_protocol_sha256": (
                phase0_evidence_module.QWEN_PROMPT_PROTOCOL_SHA256
            ),
            "runtime_identity": qwen_capability["runtime_identity"],
            "source_bundle_sha256": (
                phase0_evidence_module.QWEN_SOURCE_BUNDLE_SHA256
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return VideoVerifierRunManifest(
        schema_version=VIDEO_VERIFIER_RUN_SCHEMA_VERSION,
        run_id="phase0-video-verifier",
        created_at="2026-09-04T00:00:00Z",
        finished_at="2026-09-04T00:00:10Z",
        run_status="complete",
        code_sha=_CODE_SHA,
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        dataset_revision=video_verifier_dataset_revision(dataset),
        candidate_set_revision="c" * 64,
        verifier_identity=ComponentIdentity("video_verifier", identity),
        strict_no_fallback=True,
        attempts=attempts,
        summary=VideoVerifierSummary(
            case_count=len(attempts),
            match_count=len(attempts),
            model_miss_count=0,
            infrastructure_error_count=0,
        ),
    )


def _input_sha256s() -> dict[str, str]:
    identifiers = {
        "frozen_metric_policy",
        "product_dataset",
        "video_verifier_dataset",
        "ml_environment_attestation",
        "full_ml_smoke",
        "baseline_batch",
        "video_verifier_run",
        "rollback_proof",
        *(f"benchmark_run.{item}" for item in REQUIRED_EXECUTED_PROFILE_IDS),
    }
    return {
        identifier: sha256(identifier.encode("utf-8")).hexdigest()
        for identifier in identifiers
    }


def _baseline_batch(
    *,
    input_sha256s: dict[str, str] | None = None,
    runs: dict[str, BenchmarkRunManifest] | None = None,
) -> dict[str, object]:
    input_sha256s = input_sha256s or _input_sha256s()
    runs = runs or _benchmark_runs()
    environment = phase0_evidence_module._current_ml_environment_contract()
    policy = load_frozen_metric_policy(POLICY_PATH)
    dataset = load_dataset(PRODUCT_DATASET_PATH)
    return {
        "schema_version": 1,
        "status": "complete",
        "code_sha": _CODE_SHA,
        "dataset_revision": dataset_revision(dataset),
        "policy_revision": phase0_evidence_module.frozen_metric_policy_revision(
            policy
        ),
        "ml_environment_manifest_identity": environment["manifest_identity"],
        "ml_environment_attestation_id": environment["attestation_id"],
        "environment_bindings": dict(
            phase0_evidence_module.BASELINE_BATCH_ENVIRONMENT_BINDINGS
        ),
        "profile_runs": [
            {
                "profile_id": profile_id,
                "run_id": runs[profile_id].run_id,
                "manifest_sha256": input_sha256s[
                    f"benchmark_run.{profile_id}"
                ],
            }
            for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
        ],
        "worker_lifecycle": {
            "cleanup_status": "complete",
            "retirement_status": "complete",
        },
    }


def _write_json_fixture(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _collection_inputs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    paths = {
        "policy": tmp_path / "policy.json",
        "product": tmp_path / "product.json",
        "verifier": tmp_path / "verifier.json",
        "environment": tmp_path / "environment.json",
        "smoke": tmp_path / "smoke.json",
        "batch": tmp_path / "baseline-batch.json",
        "verifier_run": tmp_path / "verifier-run.json",
        "rollback": tmp_path / "rollback.json",
    }
    paths["policy"].write_bytes(POLICY_PATH.read_bytes())
    paths["product"].write_bytes(PRODUCT_DATASET_PATH.read_bytes())
    paths["verifier"].write_bytes(VERIFIER_DATASET_PATH.read_bytes())
    _write_json_fixture(paths["environment"], _environment_report())
    _write_json_fixture(paths["smoke"], _full_ml_smoke())
    _write_json_fixture(paths["verifier_run"], video_verifier_run_to_dict(_video_verifier_run()))
    _write_json_fixture(paths["rollback"], _rollback_proof())
    run_paths: dict[str, Path] = {}
    for profile_id, run in _benchmark_runs().items():
        path = tmp_path / f"run-{profile_id}.json"
        _write_json_fixture(path, run_to_dict(run))
        run_paths[profile_id] = path
    input_sha256s = _input_sha256s()
    for profile_id, path in run_paths.items():
        input_sha256s[f"benchmark_run.{profile_id}"] = sha256(
            path.read_bytes()
        ).hexdigest()
    _write_json_fixture(
        paths["batch"],
        _baseline_batch(input_sha256s=input_sha256s),
    )
    return paths, run_paths


def test_environment_attestation_requires_every_phase0_environment() -> None:
    summary = validate_ml_environment_attestation(_environment_report())

    assert summary["status"] == "complete"
    assert summary["host"]["chip"] == "Apple M4 Pro"  # type: ignore[index]

    drifted = _environment_report()
    drifted["environments"] = [  # type: ignore[index]
        item
        for item in drifted["environments"]  # type: ignore[union-attr]
        if item["id"] != "ocr"  # type: ignore[index]
    ]
    with pytest.raises(Phase0EvidenceError, match="environment_attestation_invalid"):
        validate_ml_environment_attestation(drifted)

    path_leak = _environment_report()
    path_leak["capabilities"][0]["runtime_identity"] = (  # type: ignore[index]
        "/Users/private/runtime"
    )
    with pytest.raises(Phase0EvidenceError, match="environment_attestation_invalid"):
        validate_ml_environment_attestation(path_leak)


@pytest.mark.parametrize(
    "mutation",
    (
        "attestation_id",
        "manifest_identity",
        "omitted_contract",
        "contract_identity",
        "capability_model",
        "capability_runtime",
        "environment_distribution",
        "uv_identity",
        "schema_version_type",
    ),
)
def test_environment_attestation_is_pinned_to_the_current_manifest(
    mutation: str,
) -> None:
    report = _environment_report()
    if mutation == "attestation_id":
        report["attestation_id"] = "foreign-attestation"
    elif mutation == "manifest_identity":
        report["manifest_identity"] = "sha256:" + "f" * 64
    elif mutation == "omitted_contract":
        report["contracts"] = report["contracts"][:-1]  # type: ignore[index]
    elif mutation == "contract_identity":
        report["contracts"][0]["identity"] = "sha256:" + "f" * 64  # type: ignore[index]
    elif mutation == "capability_model":
        report["capabilities"][0]["model_identities"] = [  # type: ignore[index]
            "foreign/model@" + "f" * 40
        ]
    elif mutation == "capability_runtime":
        report["capabilities"][0]["runtime_identity"] = "foreign-runtime@1"  # type: ignore[index]
    elif mutation == "environment_distribution":
        report["environments"][0]["distribution_identity"] = (  # type: ignore[index]
            "sha256:" + "f" * 64
        )
    elif mutation == "uv_identity":
        report["uv"]["identity"] = "sha256:" + "f" * 64  # type: ignore[index]
    else:
        report["schema_version"] = True

    with pytest.raises(Phase0EvidenceError, match="environment_attestation_invalid"):
        validate_ml_environment_attestation(report)


def test_full_ml_smoke_requires_raw_native_measurements_and_exact_profiles() -> None:
    summary = validate_full_ml_smoke(
        _full_ml_smoke(),
        code_sha=_CODE_SHA,
        memory_limit_bytes=16 * 1024**3,
    )

    assert summary["status"] == "ready"
    assert summary["schema_version"] == 2
    assert summary["profiles"]["internvideo"]["status"] == "not_configured"  # type: ignore[index]
    assert summary["process_tree"]["peak_bytes"] == 1_500  # type: ignore[index]
    assert summary["profiles"]["qwen_verification"][  # type: ignore[index]
        "component_execution"
    ] == _component_execution("qwen_verification")

    missing = _full_ml_smoke()
    del missing["product_integration"]["profiles"]["qwen_verification"]  # type: ignore[index]
    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            missing,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )

    no_raw = _full_ml_smoke()
    no_raw["resources"]["host_resources"]["raw_samples"] = []  # type: ignore[index]
    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            no_raw,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )


def test_full_ml_smoke_rejects_legacy_v1_receipt() -> None:
    report = _full_ml_smoke()
    report["schema_version"] = 1

    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            report,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "trace_schema_bool",
        "foreign_profile_identity",
        "path_profile_identity",
        "foreign_search_configuration_identity",
        "missing_field",
        "extra_field",
        "invoked_out_of_order",
        "invoked_duplicate",
        "invoked_foreign",
        "missing_count_key",
        "extra_count_key",
        "bool_count",
        "negative_count",
        "foreign_count",
    ),
)
def test_full_ml_smoke_component_execution_trace_is_exact_and_path_free(
    mutation: str,
) -> None:
    report = _full_ml_smoke()
    trace = report["product_integration"]["profiles"]["qwen_verification"][  # type: ignore[index]
        "component_execution"
    ]
    if mutation == "trace_schema_bool":
        trace["schema_version"] = True
    elif mutation == "foreign_profile_identity":
        trace["profile_identity"] = get_profile("lighthouse").identity
    elif mutation == "path_profile_identity":
        trace["profile_identity"] = "/Users/private/profile"
    elif mutation == "foreign_search_configuration_identity":
        trace["search_configuration_identity"] = (
            "evaluation-search-configuration@1:" + "f" * 64
        )
    elif mutation == "missing_field":
        del trace["component_input_counts"]
    elif mutation == "extra_field":
        trace["profile_id"] = "qwen_verification"
    elif mutation == "invoked_out_of_order":
        trace["invoked_component_ids"] = [
            "lexical_text",
            "text_vectors",
            "visual_dense",
            "temporal_refinement",
            "lighthouse",
            "qwen_verification",
        ]
    elif mutation == "invoked_duplicate":
        trace["invoked_component_ids"].append("text_vectors")  # type: ignore[union-attr]
    elif mutation == "invoked_foreign":
        trace["invoked_component_ids"][-1] = "internvideo"  # type: ignore[index]
    elif mutation == "missing_count_key":
        del trace["component_input_counts"]["text_vectors"]  # type: ignore[index]
    elif mutation == "extra_count_key":
        trace["component_output_counts"]["internvideo"] = 1  # type: ignore[index]
    elif mutation == "bool_count":
        trace["component_evidence_counts"]["text_vectors"] = True  # type: ignore[index]
    elif mutation == "negative_count":
        trace["component_input_counts"]["text_vectors"] = -1  # type: ignore[index]
    else:
        trace["component_output_counts"]["text_vectors"] = "1"  # type: ignore[index]

    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            report,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )


@pytest.mark.parametrize(
    "field",
    (
        "invoked_component_ids",
        "component_input_counts",
        "component_output_counts",
        "component_evidence_counts",
    ),
)
def test_full_ml_smoke_rejects_unselected_component_execution_leakage(
    field: str,
) -> None:
    report = _full_ml_smoke()
    trace = report["product_integration"]["profiles"]["lexical_qdrant"][  # type: ignore[index]
        "component_execution"
    ]
    if field == "invoked_component_ids":
        trace[field].append("visual_dense")  # type: ignore[union-attr]
    else:
        trace[field]["visual_dense"] = 1  # type: ignore[index]

    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            report,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )


@pytest.mark.parametrize(
    ("profile_id", "component_id"),
    (
        ("lexical_qdrant", "text_vectors"),
        ("lexical_qdrant", "lexical_text"),
        ("dense_siglip", "visual_dense"),
        ("temporal_refinement", "temporal_refinement"),
        ("lighthouse", "lighthouse"),
        ("qwen_verification", "qwen_verification"),
    ),
)
@pytest.mark.parametrize(
    "count_field",
    (
        "component_input_counts",
        "component_output_counts",
        "component_evidence_counts",
    ),
)
def test_full_ml_smoke_requires_substantive_selected_component_execution(
    profile_id: str,
    component_id: str,
    count_field: str,
) -> None:
    report = _full_ml_smoke()
    trace = report["product_integration"]["profiles"][profile_id][  # type: ignore[index]
        "component_execution"
    ]
    trace[count_field][component_id] = 0  # type: ignore[index]

    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            report,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_identity",
        "attestation_id",
        "environment_binding",
        "schema_version_type",
    ),
)
def test_full_ml_smoke_is_bound_to_the_pinned_ml_environment(
    mutation: str,
) -> None:
    report = _full_ml_smoke()
    if mutation == "manifest_identity":
        report["ml_environment_manifest_identity"] = "sha256:" + "f" * 64
    elif mutation == "attestation_id":
        report["ml_environment_attestation_id"] = "foreign-attestation"
    elif mutation == "environment_binding":
        report["environment_bindings"]["qwen"] = "base"  # type: ignore[index]
    else:
        report["schema_version"] = True

    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            report,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_identity",
        "attestation_id",
        "environment_binding",
        "run_manifest",
        "run_order",
        "cleanup",
        "schema_version_type",
    ),
)
def test_baseline_batch_binds_every_run_to_the_attested_environment(
    mutation: str,
) -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    dataset = load_dataset(PRODUCT_DATASET_PATH)
    runs = _benchmark_runs()
    input_sha256s = _input_sha256s()
    receipt = _baseline_batch(input_sha256s=input_sha256s, runs=runs)
    if mutation == "manifest_identity":
        receipt["ml_environment_manifest_identity"] = "sha256:" + "f" * 64
    elif mutation == "attestation_id":
        receipt["ml_environment_attestation_id"] = "foreign-attestation"
    elif mutation == "environment_binding":
        receipt["environment_bindings"]["qwen"] = "base"  # type: ignore[index]
    elif mutation == "run_manifest":
        receipt["profile_runs"][0]["manifest_sha256"] = "f" * 64  # type: ignore[index]
    elif mutation == "run_order":
        receipt["profile_runs"][0], receipt["profile_runs"][1] = (  # type: ignore[index]
            receipt["profile_runs"][1],  # type: ignore[index]
            receipt["profile_runs"][0],  # type: ignore[index]
        )
    elif mutation == "cleanup":
        receipt["worker_lifecycle"]["cleanup_status"] = "pending"  # type: ignore[index]
    else:
        receipt["schema_version"] = True

    environment = validate_ml_environment_attestation(_environment_report())
    with pytest.raises(Phase0EvidenceError, match="baseline_batch_invalid"):
        validate_baseline_batch(
            receipt,
            policy=policy,
            dataset=dataset,
            environment=environment,
            runs=runs,
            run_manifest_sha256s={
                profile_id: input_sha256s[f"benchmark_run.{profile_id}"]
                for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
            },
            code_sha=_CODE_SHA,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("proof_id", "hand-authored-proof"),
        ("forced_failure", "some_other_failure"),
        ("semantic_assertions", ["unreviewed_claim"]),
        ("test_node_id", "backend/tests/test_other.py::test_other"),
        ("test_result", "failed"),
        ("test_source_sha256", "0" * 64),
        ("schema_version", True),
    ),
)
def test_rollback_proof_is_fail_closed(field: str, value: object) -> None:
    proof = _rollback_proof()
    proof[field] = value

    with pytest.raises(Phase0EvidenceError, match="rollback_proof_invalid"):
        validate_rollback_proof(proof, code_sha=_CODE_SHA)


def test_full_ml_smoke_rejects_oom_swap_and_memory_breach() -> None:
    for mutation in ("oom", "swap", "rss"):
        report = deepcopy(_full_ml_smoke())
        if mutation == "oom":
            report["resources"]["oom"] = {"status": "observed"}  # type: ignore[index]
        elif mutation == "swap":
            report["resources"]["system_wide_pressure_deltas"][  # type: ignore[index]
                "swapouts_pages"
            ] = 1
        else:
            rss = report["resources"]["sampled_peak_process_tree_rss_bytes"]  # type: ignore[index]
            rss["value"] = 16 * 1024**3 + 1  # type: ignore[index]
            rss["samples_bytes"][-1] = 16 * 1024**3 + 1  # type: ignore[index]

        with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
            validate_full_ml_smoke(
                report,
                code_sha=_CODE_SHA,
                memory_limit_bytes=16 * 1024**3,
            )


@pytest.mark.parametrize("field", ("code_sha_before", "code_sha_after"))
def test_full_ml_smoke_binds_clean_code_identity(field: str) -> None:
    report = _full_ml_smoke()
    report[field] = "c" * 40

    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            report,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )


@pytest.mark.parametrize("counter", ("recovery_count", "swapins_pages", "swapouts_pages"))
def test_full_ml_smoke_rejects_counter_reset(counter: str) -> None:
    report = _full_ml_smoke()
    samples = report["resources"]["host_resources"]["raw_samples"]  # type: ignore[index]
    if counter == "recovery_count":
        samples[1]["metal"][counter] = samples[0]["metal"][counter] - 1
    else:
        samples[1]["virtual_memory"][counter] = (  # type: ignore[index]
            samples[0]["virtual_memory"][counter] - 1  # type: ignore[index]
        )

    with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
        validate_full_ml_smoke(
            report,
            code_sha=_CODE_SHA,
            memory_limit_bytes=16 * 1024**3,
        )


def test_full_ml_smoke_requires_strict_timeline_and_exact_typed_steps() -> None:
    for mutation in ("timeline", "step_set", "step_observation"):
        report = deepcopy(_full_ml_smoke())
        if mutation == "timeline":
            report["resources"]["host_resources"]["raw_samples"][1][  # type: ignore[index]
                "elapsed_nanoseconds"
            ] = 0
        elif mutation == "step_set":
            report["steps"] = report["steps"][:-1]  # type: ignore[index]
        else:
            report["steps"][0]["observations"]["dimensions"] = True  # type: ignore[index]

        with pytest.raises(Phase0EvidenceError, match="full_ml_smoke_invalid"):
            validate_full_ml_smoke(
                report,
                code_sha=_CODE_SHA,
                memory_limit_bytes=16 * 1024**3,
            )


def test_builds_complete_path_free_bundle_from_five_auditable_runs() -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    product_dataset = load_dataset(PRODUCT_DATASET_PATH)
    verifier_dataset = video_verifier_dataset_from_json(
        VERIFIER_DATASET_PATH.read_bytes()
    )

    artifacts = build_phase0_evidence(
        policy=policy,
        product_dataset=product_dataset,
        verifier_dataset=verifier_dataset,
        environment_attestation=_environment_report(),
        full_ml_smoke=_full_ml_smoke(),
        baseline_batch=_baseline_batch(),
        video_verifier_run=_video_verifier_run(),
        benchmark_runs=_benchmark_runs(),
        rollback_proof=_rollback_proof(),
        code_sha=_CODE_SHA,
        input_sha256s=_input_sha256s(),
    )

    bundle_ids = {
        artifact["bundle_id"]
        for _name, artifact in artifacts.files()
    }
    assert len(bundle_ids) == 1
    assert artifacts.baseline_snapshot["schema_version"] == 2
    assert artifacts.sanitized_report["schema_version"] == 2
    assert artifacts.baseline_snapshot["promotion_eligible"] is False
    assert [
        item["profile_id"]
        for item in artifacts.baseline_snapshot["profiles"]  # type: ignore[index]
    ] == [*REQUIRED_EXECUTED_PROFILE_IDS, "internvideo"]
    assert len(artifacts.raw_measurements["benchmark_runs"]) == 5  # type: ignore[arg-type]
    assert artifacts.error_ledger["infrastructure_error_count"] == 0
    retained_execution = artifacts.sanitized_report["full_ml_smoke"][  # type: ignore[index]
        "component_execution"
    ]
    assert retained_execution == {
        profile_id: _component_execution(profile_id)
        for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
    }
    assert artifacts.baseline_snapshot["full_ml_smoke"][  # type: ignore[index]
        "component_execution"
    ] == retained_execution
    assert artifacts.sanitized_report["full_ml_smoke"]["schema_version"] == 2  # type: ignore[index]
    for profile in artifacts.sanitized_report["profiles"]:  # type: ignore[index]
        metrics = profile["metrics"]
        assert metrics["infrastructure_error_count"] == {
            "status": "measured",
            "unit": "count",
            "value": 0,
        }
        assert metrics["slice.event_class.made_2.case_count"] == {
            "reason_code": "slice_absent_from_frozen_dataset",
            "status": "not_observed",
            "unit": "count",
        }
        assert metrics[
            "slice.event_class.made_2.candidate_recall_at_50"
        ] == {
            "reason_code": "slice_absent_from_frozen_dataset",
            "status": "not_observed",
            "unit": "fraction",
        }
    encoded = json.dumps(
        dict(artifacts.files()),
        ensure_ascii=False,
        sort_keys=True,
    )
    assert str(PROJECT_ROOT) not in encoded
    assert "/Users/" not in encoded
    assert "prepared_input" not in encoded
    assert product_dataset.cases[0].query not in encoded


def test_benchmark_gate_allows_profile_specific_runtime_and_measurement_digests() -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    dataset = load_dataset(PRODUCT_DATASET_PATH)
    runs = _benchmark_runs()
    qwen = runs["qwen_verification"]
    runs["qwen_verification"] = replace(
        qwen,
        measurement_protocol=replace(
            qwen.measurement_protocol,
            identity="process-tree-rss-50ms-contained-storage@2:" + "8" * 64,
        ),
    )

    summaries, measurements, errors = validate_benchmark_runs(
        policy,
        dataset,
        runs,
        code_sha=_CODE_SHA,
    )
    assert len(summaries) == len(measurements) == len(errors) == 5

    missing = dict(runs)
    del missing["qwen_verification"]
    with pytest.raises(Phase0EvidenceError, match="benchmark_profile_set_invalid"):
        validate_benchmark_runs(policy, dataset, missing, code_sha=_CODE_SHA)

    no_raw = dict(runs)
    no_raw["qwen_verification"] = replace(
        no_raw["qwen_verification"],
        measurement_evidence_status="legacy_unavailable",
        measurement_evidence=None,
    )
    with pytest.raises(Phase0EvidenceError, match="benchmark_run_invalid"):
        validate_benchmark_runs(policy, dataset, no_raw, code_sha=_CODE_SHA)


def test_benchmark_gate_rejects_drift_in_a_shared_index_identity() -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    dataset = load_dataset(PRODUCT_DATASET_PATH)
    runs = _benchmark_runs()
    qwen = runs["qwen_verification"]
    runs["qwen_verification"] = replace(
        qwen,
        index_identities=tuple(
            replace(item, identity="e" * 64)
            if item.component_id == "text_vector_index"
            else item
            for item in qwen.index_identities
        ),
    )

    with pytest.raises(Phase0EvidenceError, match="benchmark_comparability_invalid"):
        validate_benchmark_runs(policy, dataset, runs, code_sha=_CODE_SHA)


def test_benchmark_gate_requires_a_digest_sealed_measurement_protocol() -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    dataset = load_dataset(PRODUCT_DATASET_PATH)
    runs = _benchmark_runs()
    lexical = runs["lexical_qdrant"]
    runs["lexical_qdrant"] = replace(
        lexical,
        measurement_protocol=replace(
            lexical.measurement_protocol,
            identity="process-tree-rss-50ms-contained-storage@2:/Users/private",
        ),
    )

    with pytest.raises(
        Phase0EvidenceError,
        match="benchmark_measurement_protocol_invalid",
    ):
        validate_benchmark_runs(policy, dataset, runs, code_sha=_CODE_SHA)


def test_video_verifier_status_cannot_disagree_with_its_raw_prediction() -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    product_dataset = load_dataset(PRODUCT_DATASET_PATH)
    verifier_dataset = video_verifier_dataset_from_json(
        VERIFIER_DATASET_PATH.read_bytes()
    )
    run = _video_verifier_run()
    first = run.attempts[0]
    prediction = first.prediction
    assert prediction is not None
    mismatched_facts = tuple(
        replace(fact, expected=not fact.expected)
        if fact.expected is not None
        else fact
        for fact in prediction.facts
    )
    attempts = (
        replace(
            first,
            prediction=replace(prediction, facts=mismatched_facts),
            status="match",
        ),
        *run.attempts[1:],
    )
    dishonest = replace(
        run,
        attempts=attempts,
        summary=VideoVerifierSummary(
            case_count=len(attempts),
            match_count=len(attempts),
            model_miss_count=0,
            infrastructure_error_count=0,
        ),
    )

    with pytest.raises(Phase0EvidenceError, match="video_verifier_evidence_invalid"):
        build_phase0_evidence(
            policy=policy,
            product_dataset=product_dataset,
            verifier_dataset=verifier_dataset,
            environment_attestation=_environment_report(),
            full_ml_smoke=_full_ml_smoke(),
            baseline_batch=_baseline_batch(),
            video_verifier_run=dishonest,
            benchmark_runs=_benchmark_runs(),
            rollback_proof=_rollback_proof(),
            code_sha=_CODE_SHA,
            input_sha256s=_input_sha256s(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mode", "worker"),
        ("contract", "foreign-worker-v1"),
        ("source_bundle_sha256", "f" * 64),
        ("prompt_protocol_sha256", "f" * 64),
        ("runtime_identity", "foreign-runtime@1"),
        ("fps", 8.1),
    ),
)
def test_video_verifier_identity_is_pinned_to_the_current_worker(
    field: str,
    value: object,
) -> None:
    policy = load_frozen_metric_policy(POLICY_PATH)
    verifier_dataset = video_verifier_dataset_from_json(
        VERIFIER_DATASET_PATH.read_bytes()
    )
    run = _video_verifier_run()
    identity = json.loads(run.verifier_identity.identity)
    identity[field] = value
    run = replace(
        run,
        verifier_identity=ComponentIdentity(
            "video_verifier",
            json.dumps(identity, sort_keys=True, separators=(",", ":")),
        ),
    )

    with pytest.raises(Phase0EvidenceError, match="video_verifier_evidence_invalid"):
        build_phase0_evidence(
            policy=policy,
            product_dataset=load_dataset(PRODUCT_DATASET_PATH),
            verifier_dataset=verifier_dataset,
            environment_attestation=_environment_report(),
            full_ml_smoke=_full_ml_smoke(),
            baseline_batch=_baseline_batch(),
            video_verifier_run=run,
            benchmark_runs=_benchmark_runs(),
            rollback_proof=_rollback_proof(),
            code_sha=_CODE_SHA,
            input_sha256s=_input_sha256s(),
        )


@pytest.mark.parametrize(
    "swapped_input",
    ("policy", "product", "verifier", "verifier_run", "batch"),
)
def test_collector_parses_the_exact_bytes_it_hashes_under_input_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swapped_input: str,
) -> None:
    paths, run_paths = _collection_inputs(tmp_path)
    original_load = phase0_evidence_module._load_json
    swapped = False

    def load_then_swap(path: Path, maximum: int, context: str):  # type: ignore[no-untyped-def]
        nonlocal swapped
        loaded = original_load(path, maximum, context)
        if not swapped and Path(path) == paths[swapped_input]:
            paths[swapped_input].write_text("{}", encoding="utf-8")
            swapped = True
        return loaded

    monkeypatch.setattr(phase0_evidence_module, "_load_json", load_then_swap)
    output = tmp_path / "bundle"

    artifacts = collect_phase0_evidence(
        policy_path=paths["policy"],
        product_dataset_path=paths["product"],
        verifier_dataset_path=paths["verifier"],
        environment_attestation_path=paths["environment"],
        full_ml_smoke_path=paths["smoke"],
        baseline_batch_path=paths["batch"],
        video_verifier_run_path=paths["verifier_run"],
        benchmark_run_paths=run_paths,
        rollback_proof_path=paths["rollback"],
        output_dir=output,
        code_sha=_CODE_SHA,
        code_identity_resolver=lambda: _CODE_SHA,
    )

    assert swapped is True
    assert output.is_dir()
    assert sorted(path.name for path in output.iterdir()) == [
        "baseline-snapshot.json",
        "error-ledger.json",
        "raw-measurements.json",
        "sanitized-report.json",
    ]
    assert artifacts.baseline_snapshot["evidence_status"] == "valid"

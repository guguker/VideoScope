"""Replay the retained diagnostic checks; never run a model or accept a bundle."""

from hashlib import sha256
import json
from pathlib import Path

from videoscope.benchmark.phase0_evidence import (
    Phase0EvidenceError,
    REQUIRED_EXECUTED_PROFILE_IDS,
    _validate_host_resources,
    _validate_product_dataset_binding,
    _validate_video_verifier,
    dataset_from_dict,
    frozen_metric_policy_from_dict,
    run_from_dict,
    validate_baseline_batch,
    validate_benchmark_runs,
    validate_full_ml_smoke,
    validate_ml_environment_attestation,
    validate_rollback_proof,
    video_verifier_dataset_from_json,
    video_verifier_run_from_dict,
)


ROOT = Path(__file__).resolve().parent
CODE_SHA = "7054f137f92c18676461242a18c8fc2619898b68"


def read(name):
    return json.loads((ROOT / name).read_bytes())


record = read("control-record.json")
assert record["code_sha"] == CODE_SHA
for name, binding in record["bindings"].items():
    assert Path(name).name == name
    raw = (ROOT / name).read_bytes()
    assert len(raw) == binding["byte_size"]
    assert sha256(raw).hexdigest() == binding["sha256"]

environment = validate_ml_environment_attestation(read("ml-environment.json"))
validate_rollback_proof(read("phase0-rollback-proof.json"), code_sha=CODE_SHA)

# Validate the other inputs independently: the unchanged full collector stops
# at the rejected smoke before it reaches these checks.
benchmarks = ROOT.parents[2]
policy = frozen_metric_policy_from_dict(json.loads(
    (benchmarks / "policies/phase0-regression-v1.json").read_bytes(),
))
dataset = dataset_from_dict(json.loads(
    (benchmarks / "product-retrieval/seed-v1.json").read_bytes(),
))
verifier_dataset = video_verifier_dataset_from_json(
    (benchmarks / "video-verifier/seed-v1.json").read_text(),
)
_validate_product_dataset_binding(policy, dataset)
runs = {
    profile_id: run_from_dict(read(f"{profile_id}-manifest.json"))
    for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
}
validate_benchmark_runs(policy, dataset, runs, code_sha=CODE_SHA)
validate_baseline_batch(
    read("phase0-baseline-receipt.json"), policy=policy, dataset=dataset,
    environment=environment, runs=runs, code_sha=CODE_SHA,
    run_manifest_sha256s={
        profile_id: record["bindings"][f"{profile_id}-manifest.json"]["sha256"]
        for profile_id in REQUIRED_EXECUTED_PROFILE_IDS
    },
)
capability = next(
    item for item in environment["capabilities"]
    if item["id"] == "qwen_verification"
)
direct = _validate_video_verifier(
    policy, verifier_dataset,
    video_verifier_run_from_dict(read("video-verifier-run.json")),
    code_sha=CODE_SHA,
    expected_model_identities=capability["model_identities"],
    expected_runtime_identity=capability["runtime_identity"],
)
assert direct["summary"] == {
    "case_count": 10, "match_count": 3, "model_miss_count": 7,
    "infrastructure_error_count": 0,
}
smoke = read("full-ml-smoke-diagnostic.json")
trace = read("full-ml-timeline.json")
assert smoke["code_sha_before"] == smoke["code_sha_after"] == CODE_SHA
assert trace["code_sha"] == CODE_SHA
assert smoke["status"] == "ready" and smoke["workspace_cleanup"] == "complete"
assert trace["status"] == "complete" and trace["failure_code"] is None
assert trace["smoke_exit_code"] == 0
assert record["bindings"]["full-ml-smoke-diagnostic.json"] == {
    "sha256": trace["smoke_stdout_sha256"],
    "byte_size": trace["smoke_stdout_byte_size"],
}
assert len(smoke["steps"]) == 8
assert all(step["status"] == "complete" for step in smoke["steps"])
profiles = smoke["product_integration"]["profiles"]
assert profiles["internvideo"]["status"] == "not_configured"
assert sum(p["status"] == "complete" for p in profiles.values()) == 5


def verify_timeline(timeline, samples):
    rows = timeline["process_samples"]
    assert [row["total_rss_bytes"] for row in rows] == samples
    identities = {}
    for row in rows:
        assert sum(role["rss_bytes"] for role in row["roles"]) == row["total_rss_bytes"]
        for role in row["roles"]:
            identities.setdefault(role["role"], set()).add(role["identity_sha256"])
    assert all(len(values) == 1 for values in identities.values())
    for before, after in zip(rows, rows[1:]):
        assert before["snapshot_finished_elapsed_ns"] <= after["snapshot_started_elapsed_ns"]
    stack = []
    for event in timeline["events"]:
        name, phase = event["id"].rsplit(".", 1)
        if phase == "begin":
            stack.append(name)
        else:
            assert phase == "end" and stack.pop() == name
    assert not stack


verify_timeline(trace, smoke["resources"]["sampled_peak_process_tree_rss_bytes"]["samples_bytes"])
assert len(trace["events"]) == 70 and len(trace["process_samples"]) == 4808
_validate_host_resources(smoke["resources"]["host_resources"])
assert smoke["resources"]["system_wide_pressure_deltas"] == {
    "swapins_pages": 176, "swapins_bytes": 2883584,
    "swapouts_pages": 0, "swapouts_bytes": 0, "metal_recovery_count": 0,
}
try:
    validate_full_ml_smoke(smoke, code_sha=CODE_SHA, memory_limit_bytes=17179869184)
except Phase0EvidenceError as error:
    assert error.code == "full_ml_smoke_invalid"
else:
    raise AssertionError("The unchanged resource gate unexpectedly accepted the smoke")

idle = read("no-ml-control.json")
assert idle["status"] == "complete" and idle["samplers_closed"] is True
assert idle["code_sha"] == CODE_SHA
assert idle["managed_workers"] == idle["inference_calls"] == 0
assert idle["worker_count_before"] == idle["worker_count_after"] == 0
assert idle["timeline"]["status"] == "complete"
assert idle["timeline"]["failure_code"] is None
verify_timeline(idle["timeline"], idle["process_tree"]["samples_bytes"])
assert len(idle["timeline"]["process_samples"]) == 2088
assert all(
    len(row["roles"]) == 1
    and row["roles"][0]["role"] == "owner"
    and row["roles"][0]["process_count"] == 1
    for row in idle["timeline"]["process_samples"]
)
host = _validate_host_resources(idle["host_resources"])
assert host["virtual_memory"]["swapins_delta_pages"] == 4
assert host["virtual_memory"]["swapouts_delta_pages"] == 0
assert host["metal"]["recovery_delta"] == 0
assert read("no-ml-control-invalid.json")["raw_measurements_available"] is False
print(json.dumps({
    "status": "verified_negative_control",
    "code_sha": CODE_SHA,
    "bound_files": len(record["bindings"]),
    "smoke_gate": "rejected",
    "smoke_swapins_pages": 176,
    "no_ml_swapins_pages": 4,
    "all_non_smoke_validators": "passed",
    "phase0_complete": False,
    "accepted_bundle_created": False,
}, sort_keys=True))

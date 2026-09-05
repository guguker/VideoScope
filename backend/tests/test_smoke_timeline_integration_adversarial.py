from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_full_ml_smoke import (
    _ResourceMonitor, _dependencies, _load_script, _models_root,
    _offline_environment,
)


class _FailingTrace:
    def __init__(self, trigger: str, *, fail_after_trigger: bool = False) -> None:
        self.trigger = trigger
        self.fail_after_trigger = fail_after_trigger
        self.triggered = False

    def record_event(self, event_id: str) -> None:
        if event_id == self.trigger:
            self.triggered = True
        if event_id == self.trigger or (self.triggered and self.fail_after_trigger):
            raise RuntimeError("secret model output /private/user/data bearer-token")


@pytest.mark.parametrize("trigger", [
    "workers.start.end", "monitor.start.end", "cleanup.ocr.begin",
    "monitor.finish.begin", "monitor.close.begin", "workers.close.begin",
    "workspace.cleanup.begin",
])
def test_timeline_failure_never_loses_acquired_ownership_or_skips_cleanup(
    tmp_path: Path, trigger: str,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models = _models_root(tmp_path)
    events: list[str] = []

    class RecordingMonitor(_ResourceMonitor):
        def attach_diagnostic(self, _timeline: object) -> None:
            pass

        def finish(self):  # type: ignore[no-untyped-def]
            events.append("monitor.finish")
            return super().finish()

        def close(self) -> None:
            events.append("monitor.close")
            super().close()

    monitor = RecordingMonitor()
    dependencies = _dependencies(script, events)._replace(
        build_resource_monitor=lambda _workers: monitor,
    )
    trace = _FailingTrace(trigger, fail_after_trigger=True)
    token = script._DIAGNOSTIC_TIMELINE.set(trace)
    try:
        try:
            result = script.execute(
                root, models_root=models, environ=_offline_environment(root, models),
                dependencies=dependencies,
            )
        except script.SmokeError:
            result = None
        assert trace.triggered
        assert events.count("workers.close") == 1
        assert events.count("monitor.finish") == 1
        assert events.count("monitor.close") == 1
        assert events.count("ocr.close") == 1
        assert monitor.started and monitor.finished
        assert list(root.iterdir()) == []
        assert script._DIAGNOSTIC_FAILURE.get() is True
        if result is not None:
            assert result["status"] == "ready"
            assert "secret model output" not in json.dumps(result)
    finally:
        script._DIAGNOSTIC_TIMELINE.reset(token)


def test_latched_timeline_failure_preserves_primary_inference_error_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_script()
    root = tmp_path / "smoke"
    root.mkdir(mode=0o700)
    models = _models_root(tmp_path)
    events: list[str] = []
    monkeypatch.setattr(_ResourceMonitor, "attach_diagnostic", lambda *_args: None, raising=False)
    trace = _FailingTrace("cleanup.ocr.begin", fail_after_trigger=True)
    token = script._DIAGNOSTIC_TIMELINE.set(trace)
    try:
        with pytest.raises(script.SmokeInfrastructureError) as caught:
            script.execute(
                root, models_root=models, environ=_offline_environment(root, models),
                dependencies=_dependencies(script, events, fail_image=True),
            )
        assert caught.value.code == "image_embedding_failed"
        assert caught.value.component == "vision"
        assert "ocr.close" in events
        assert "workers.close" in events
        assert list(root.iterdir()) == []
        assert script._DIAGNOSTIC_FAILURE.get() is True
        encoded = json.dumps(script._error_payload(caught.value, include_diagnostics=True))
        assert "secret model output" not in encoded
        assert "/private/user/data" not in encoded
    finally:
        script._DIAGNOSTIC_TIMELINE.reset(token)


def test_native_monitor_binds_real_timeline_epoch_to_existing_samplers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from videoscope.benchmark.host_resources import HostResourceSnapshot
    from videoscope.benchmark.measurements import ManagedProcessBinding, ProcessRecord
    from videoscope.benchmark.smoke_timeline import SmokeTimeline

    script = _load_script()
    root_pid = os.getpid()
    records = (ProcessRecord(root_pid, 1, 100, "owner-start", "owner-python"),) + tuple(
        ProcessRecord(root_pid + index, root_pid, index * 10, f"{role}-start", f"{role}-python")
        for index, role in enumerate(script._MANAGED_WORKER_ROLES, 1)
    )
    bindings = tuple(
        ManagedProcessBinding(record.pid, record.start_token, record.executable_identity, role)
        for record, role in zip(records[1:], script._MANAGED_WORKER_ROLES)
    )
    process_calls: list[bool] = []
    host_calls: list[bool] = []
    monkeypatch.setattr(script, "create_native_process_snapshot_provider", lambda **_kwargs: SimpleNamespace(
        identity="synthetic-native-process@1",
        snapshot=lambda: (process_calls.append(True) or records),
    ))
    monkeypatch.setattr(script, "create_host_resource_snapshot_provider", lambda: SimpleNamespace(
        identity="synthetic-native-host@1", scope="system_wide",
        snapshot=lambda: (host_calls.append(True) or HostResourceSnapshot(1, 2, 0, 0, 0, 16384)),
    ))
    timeline = SmokeTimeline(script._DIAGNOSTIC_EVENT_IDS)
    monitor = script.build_resource_monitor(bindings)
    monitor.attach_diagnostic(timeline)
    try:
        monitor.start()
        measurement = monitor.finish()
    finally:
        monitor.close()
    result = timeline.to_portable_dict(code_sha="a" * 40, status="complete")
    assert result["host_sampler_started_elapsed_ns"] >= 0
    assert result["host_sampler_started_elapsed_ns"] <= result["process_samples"][0]["snapshot_started_elapsed_ns"]
    assert len(result["process_samples"]) == len(process_calls)
    assert len(measurement["host_resources"].raw_samples) == len(host_calls)
    assert [sample["total_rss_bytes"] for sample in result["process_samples"]] == list(measurement["process_tree"].samples_bytes)

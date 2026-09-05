from __future__ import annotations

from contextvars import ContextVar
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_SHA = "a" * 40
PRIVATE_ERROR = "/private/model/location secret-token hf_private_token"


@pytest.fixture
def diagnostic(monkeypatch, tmp_path):
    path = PROJECT_ROOT / "scripts" / "full-ml-smoke-diagnostic.py"
    spec = importlib.util.spec_from_file_location("videoscope_test_smoke_diagnostic", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temporary = tmp_path.resolve()
    roots = {}
    for name in ("smoke", "models", "home", "hf", "ocr", "receipts"):
        roots[name] = temporary / name
        roots[name].mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(roots["home"]))
    monkeypatch.setenv("HF_HOME", str(roots["hf"]))
    monkeypatch.setenv("VIDEOSCOPE_OCR_MODEL_ROOT", str(roots["ocr"]))
    timeline_path = roots["receipts"] / "timeline.json"
    context = ContextVar("test_diagnostic", default=None)
    failure_context = ContextVar("test_diagnostic_failure", default=False)
    timeline_calls = []

    class Timeline:
        def __init__(self, event_ids):
            assert event_ids == frozenset({"start", "end"})

        def to_portable_dict(self, *, code_sha, status):
            timeline_calls.append((code_sha, status))
            return {
                "schema_version": 1,
                "code_sha": code_sha,
                "status": status,
                "events": [],
                "process_samples": [],
                "attribution": "temporal_association_not_causality",
                "measurement_use": "diagnostic_only_not_gate_evidence",
            }

    state = SimpleNamespace(
        module=module, roots=roots, timeline_path=timeline_path,
        context=context, failure_context=failure_context,
        timeline_calls=timeline_calls, calls=[], observer_failed=False,
        load_smoke=module._load_smoke, write_as_text=False,
        exit_code=0, exception=None,
        stdout=(json.dumps({
            "schema_version": 2, "status": "ready",
            "code_sha_before": CODE_SHA, "code_sha_after": CODE_SHA,
        }, indent=2) + "\n").encode(),
    )

    def main(argv):
        state.calls.append(argv)
        assert isinstance(context.get(), Timeline)
        assert timeline_path.exists()
        assert stat.S_IMODE(timeline_path.stat().st_mode) == 0o600
        assert timeline_path.read_bytes() == b""
        assert failure_context.get() is False
        if state.observer_failed:
            failure_context.set(True)
        if state.exception is not None:
            raise state.exception
        if state.write_as_text:
            sys.stdout.write(state.stdout.decode("utf-8"))
        else:
            sys.stdout.buffer.write(state.stdout)
        sys.stderr.write("native compact error\n" if state.exit_code else "")
        return state.exit_code

    native = SimpleNamespace(
        main=main, _DIAGNOSTIC_TIMELINE=context,
        _DIAGNOSTIC_FAILURE=failure_context,
        _DIAGNOSTIC_EVENT_IDS=frozenset({"start", "end"}),
    )
    monkeypatch.setattr(module, "_load_smoke", lambda: native)
    monkeypatch.setattr(module, "SmokeTimeline", Timeline)
    state.argv = [
        "--timeline", str(timeline_path), "--root", str(roots["smoke"]),
        "--models-root", str(roots["models"]),
    ]
    return state


def _sidecar(state):
    return json.loads(state.timeline_path.read_bytes())


def test_ready_preserves_exact_stdout_arguments_context_and_create_once(diagnostic, capsys):
    state = diagnostic
    prior = object()
    token = state.context.set(prior)
    try:
        assert state.module.main(state.argv) == 0
        assert state.context.get() is prior
    finally:
        state.context.reset(token)
    output = capsys.readouterr()
    assert output.out.encode() == state.stdout
    assert output.err == ""
    assert state.calls == [["--root", str(state.roots["smoke"]), "--models-root", str(state.roots["models"])]]
    sidecar = _sidecar(state)
    assert sidecar["smoke_stdout_sha256"] == sha256(state.stdout).hexdigest()
    assert sidecar["smoke_stdout_byte_size"] == len(state.stdout)
    assert sidecar["smoke_exit_code"] == 0
    assert sidecar["status"] == "complete"
    assert sidecar["promotion_eligible"] is False
    assert state.timeline_calls == [(CODE_SHA, "complete")]
    original = state.timeline_path.read_bytes()
    assert state.module.main(state.argv) == 2
    assert len(state.calls) == 1
    assert state.timeline_path.read_bytes() == original


@pytest.mark.parametrize("exit_code", [2, 3, 4, 6, 70, 130])
def test_native_failure_exit_and_stdout_are_preserved(diagnostic, capsys, exit_code):
    state = diagnostic
    state.exit_code = exit_code
    state.stdout = (json.dumps({
        "schema_version": 2, "status": "failed",
        "error": {"kind": "infrastructure", "code": "judge_failed"},
        "diagnostics": {"code_sha_before": CODE_SHA},
    }) + "\n").encode()
    assert state.module.main(state.argv) == exit_code
    output = capsys.readouterr()
    assert output.out.encode() == state.stdout
    assert output.err == "native compact error\n"
    assert _sidecar(state)["status"] == "failed"
    assert state.timeline_calls == [(CODE_SHA, "failed")]
    assert state.context.get() is None


@pytest.mark.parametrize("code_sha", [None, "x" * 40, "a" * 64, 10])
def test_missing_identity_retains_primary_failure_without_fabricating_sha(diagnostic, capsys, code_sha):
    state = diagnostic
    state.exit_code = 3
    state.stdout = json.dumps({
        "schema_version": 2, "status": "failed",
        "diagnostics": {"code_sha_before": code_sha},
    }).encode()
    assert state.module.main(state.argv) == 3
    output = capsys.readouterr()
    assert output.out.encode() == state.stdout
    assert "code_identity_unavailable" in output.err
    sidecar = _sidecar(state)
    assert "code_sha" not in sidecar
    assert sidecar["status"] == "failed"
    assert sidecar["smoke_stdout_sha256"] == sha256(state.stdout).hexdigest()
    assert state.timeline_calls == []


@pytest.mark.parametrize("root_name", ["smoke", "models", "home", "hf", "ocr"])
def test_protected_roots_reject_before_loading_smoke(diagnostic, monkeypatch, capsys, root_name):
    state = diagnostic
    target = state.roots[root_name] / "timeline.json"
    state.argv[1] = str(target)
    monkeypatch.setattr(state.module, "_load_smoke", lambda: pytest.fail("loaded before path validation"))
    assert state.module.main(state.argv) == 2
    assert not target.exists()
    assert state.calls == []
    output = capsys.readouterr()
    assert str(target) not in output.err
    assert output.out == ""


@pytest.mark.parametrize("kind", ["relative", "parent_mode", "parent_symlink", "ancestor_symlink", "existing_file", "target_symlink", "foreign_owner", "checkout"])
def test_unsafe_output_path_fails_before_inference(diagnostic, monkeypatch, capsys, kind):
    state = diagnostic
    target = state.timeline_path
    existing = b"user artifact stays exact"
    if kind == "relative":
        state.argv[1] = "timeline.json"
    elif kind == "parent_mode":
        target.parent.chmod(0o755)
    elif kind in {"parent_symlink", "ancestor_symlink"}:
        link = target.parent.parent / "receipt-alias"
        link.symlink_to(target.parent, target_is_directory=True)
        if kind == "ancestor_symlink":
            (target.parent / "nested").mkdir(mode=0o700)
            state.argv[1] = str(link / "nested" / target.name)
        else:
            state.argv[1] = str(link / target.name)
    elif kind == "existing_file":
        target.write_bytes(existing)
    elif kind == "target_symlink":
        other = target.parent / "user.json"
        other.write_bytes(existing)
        target.symlink_to(other)
    elif kind == "foreign_owner":
        uid = os.getuid()
        monkeypatch.setattr(state.module.os, "getuid", lambda: uid + 1)
    elif kind == "checkout":
        state.argv[1] = str(PROJECT_ROOT / "should-not-exist-timeline.json")
    assert state.module.main(state.argv) == 2
    assert state.calls == []
    if kind in {"existing_file", "target_symlink"}:
        assert target.read_bytes() == existing
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("exit_code", [0, 3])
def test_sidecar_failure_closes_descriptor_and_preserves_primary_error(diagnostic, monkeypatch, capsys, exit_code):
    state = diagnostic
    state.exit_code = exit_code
    if exit_code:
        state.stdout = json.dumps({"schema_version": 2, "status": "failed", "diagnostics": {"code_sha_before": CODE_SHA}}).encode()
    descriptors = []

    def broken_write(fd, payload):
        descriptors.append(fd)
        raise OSError(PRIVATE_ERROR)

    monkeypatch.setattr(state.module, "_write_sidecar", broken_write)
    assert state.module.main(state.argv) == (exit_code or 70)
    output = capsys.readouterr()
    assert PRIVATE_ERROR not in output.out + output.err
    assert "diagnostic_publication_failed" in output.err
    assert state.context.get() is None
    assert state.timeline_path.exists()
    assert state.timeline_path.read_bytes() == b""
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.parametrize("exception", [RuntimeError(PRIVATE_ERROR), KeyboardInterrupt(PRIVATE_ERROR)])
def test_unexpected_native_exception_is_private_and_resets_context(diagnostic, capsys, exception):
    state = diagnostic
    state.exception = exception
    assert state.module.main(state.argv) == (130 if isinstance(exception, KeyboardInterrupt) else 70)
    output = capsys.readouterr()
    assert output.out == ""
    assert PRIVATE_ERROR not in output.err + state.timeline_path.read_text()
    assert _sidecar(state)["status"] == "failed"
    assert state.context.get() is None


def test_bounded_stdout_rejects_overflow_without_printing_partial_text(diagnostic, monkeypatch, capsys):
    state = diagnostic
    monkeypatch.setattr(state.module, "MAX_STDOUT_BYTES", 64)
    state.stdout = PRIVATE_ERROR.encode() * 10
    assert state.module.main(state.argv) == 70
    output = capsys.readouterr()
    assert output.out == ""
    assert PRIVATE_ERROR not in output.err + state.timeline_path.read_text()
    assert "stdout_limit_exceeded" in output.err
    assert _sidecar(state)["status"] == "failed"


def test_malformed_stdout_retains_primary_exit_without_leaking_text(diagnostic, capsys):
    state = diagnostic
    state.exit_code = 3
    state.stdout = PRIVATE_ERROR.encode()
    assert state.module.main(state.argv) == 3
    output = capsys.readouterr()
    assert output.out == ""
    assert PRIVATE_ERROR not in output.err + state.timeline_path.read_text()
    assert "smoke_stdout_invalid" in output.err


@pytest.mark.parametrize("exit_code", [0, 3])
def test_observer_failure_preserves_smoke_but_fails_diagnostic(diagnostic, capsys, exit_code):
    state = diagnostic
    state.observer_failed = True
    state.exit_code = exit_code
    if exit_code:
        state.stdout = json.dumps({"schema_version": 2, "status": "failed", "diagnostics": {"code_sha_before": CODE_SHA}}).encode()
    prior = state.failure_context.set(True)
    try:
        assert state.module.main(state.argv) == (exit_code or 70)
        assert state.failure_context.get() is True
    finally:
        state.failure_context.reset(prior)
    output = capsys.readouterr()
    assert output.out.encode() == state.stdout
    assert "diagnostic_observer_failed" in output.err
    assert _sidecar(state)["status"] == "failed"
    assert _sidecar(state)["smoke_exit_code"] == exit_code
    assert state.timeline_calls == [(CODE_SHA, "failed")]


def test_make_diagnostic_uses_same_environment_and_fixed_script():
    normal = subprocess.run(["make", "-n", "full-ml-smoke"], cwd=PROJECT_ROOT, check=True, text=True, capture_output=True).stdout
    diagnostic = subprocess.run(["make", "-n", "full-ml-smoke-diagnostic"], cwd=PROJECT_ROOT, check=True, text=True, capture_output=True).stdout
    assert "scripts/full-ml-smoke-diagnostic.py" in diagnostic
    assert '--timeline "$VIDEOSCOPE_FULL_ML_SMOKE_TIMELINE"' in diagnostic
    assert "VIDEOSCOPE_FULL_ML_SMOKE_TIMELINE must" in diagnostic
    assert "VIDEOSCOPE_FULL_ML_SMOKE_TIMELINE" not in normal
    for line in normal.splitlines():
        if rekey := (line.strip() if line.startswith("\t") else ""):
            if "full-ml-smoke.py" not in rekey and "--models-root" not in rekey:
                assert line in diagnostic


def test_utf8_text_stdout_binding_is_exact(diagnostic, capsys):
    state = diagnostic
    state.write_as_text = True
    state.stdout = (json.dumps({"schema_version": 2, "status": "ready", "code_sha_before": CODE_SHA, "note": "публичный контроль"}, ensure_ascii=False) + "\n").encode()
    assert state.module.main(state.argv) == 0
    assert capsys.readouterr().out.encode() == state.stdout
    assert _sidecar(state)["smoke_stdout_sha256"] == sha256(state.stdout).hexdigest()


def test_capture_setup_failure_closes_reserved_descriptor(diagnostic, monkeypatch, capsys):
    state = diagnostic
    reserve = state.module._reserve_sidecar
    descriptors = []

    def tracked_reserve(*args, **kwargs):
        fd = reserve(*args, **kwargs)
        descriptors.append(fd)
        return fd

    def broken_capture():
        raise MemoryError(PRIVATE_ERROR)

    monkeypatch.setattr(state.module, "_reserve_sidecar", tracked_reserve)
    monkeypatch.setattr(state.module, "_CapturedStdout", broken_capture)
    assert state.module.main(state.argv) == 70
    assert state.calls == []
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    output = capsys.readouterr()
    assert PRIVATE_ERROR not in output.err + state.timeline_path.read_text()
    assert output.out == ""
    assert _sidecar(state)["status"] == "failed"


@pytest.mark.parametrize("exit_code", [0, 3])
def test_timeline_export_failure_is_sanitized_and_preserves_primary(diagnostic, monkeypatch, capsys, exit_code):
    state = diagnostic
    state.exit_code = exit_code
    if exit_code:
        state.stdout = json.dumps({"schema_version": 2, "status": "failed", "diagnostics": {"code_sha_before": CODE_SHA}}).encode()

    def broken_export(*args, **kwargs):
        raise ValueError(PRIVATE_ERROR)

    monkeypatch.setattr(state.module.SmokeTimeline, "to_portable_dict", broken_export)
    assert state.module.main(state.argv) == (exit_code or 70)
    output = capsys.readouterr()
    assert PRIVATE_ERROR not in output.err + state.timeline_path.read_text()
    assert output.out.encode() == state.stdout
    assert _sidecar(state)["status"] == "failed"
    assert _sidecar(state)["code_sha"] == CODE_SHA
    assert "diagnostic_timeline_unavailable" in output.err


def test_real_smoke_main_receives_context_and_same_environment_without_inference(diagnostic, monkeypatch, capsys):
    state = diagnostic
    smoke = state.load_smoke()
    event_ids = smoke._DIAGNOSTIC_EVENT_IDS
    assert type(event_ids) is frozenset and event_ids
    monkeypatch.setattr(smoke, "_DIAGNOSTIC_EVENT_IDS", frozenset({"start", "end"}))

    def execute(root, *, models_root, environ):
        assert root == state.roots["smoke"]
        assert models_root == state.roots["models"]
        assert environ is os.environ
        assert smoke._DIAGNOSTIC_TIMELINE.get() is not None
        assert smoke._DIAGNOSTIC_FAILURE.get() is False
        return {"schema_version": 2, "status": "ready", "code_sha_before": CODE_SHA}

    monkeypatch.setattr(smoke, "execute", execute)
    monkeypatch.setattr(state.module, "_load_smoke", lambda: smoke)
    assert state.module.main(state.argv) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "ready"
    assert output.err == ""
    assert _sidecar(state)["smoke_stdout_sha256"] == sha256(output.out.encode()).hexdigest()
    assert smoke._DIAGNOSTIC_TIMELINE.get() is None
    assert smoke._DIAGNOSTIC_FAILURE.get() is False


def test_protected_root_resolution_error_never_exposes_private_path(diagnostic, capsys):
    state = diagnostic
    loop = state.roots["receipts"] / "private-secret-root"
    loop.symlink_to(loop)
    state.argv[-1] = str(loop)
    assert state.module.main(state.argv) == 2
    output = capsys.readouterr()
    assert str(loop) not in output.err
    assert output.out == ""
    assert state.calls == []
    assert not state.timeline_path.exists()


def test_unknown_argument_never_echoes_private_value(diagnostic, capsys):
    state = diagnostic
    assert state.module.main([*state.argv, "--unknown", PRIVATE_ERROR]) == 2
    output = capsys.readouterr()
    assert PRIVATE_ERROR not in output.err
    assert output.out == ""
    assert state.calls == []
    assert not state.timeline_path.exists()

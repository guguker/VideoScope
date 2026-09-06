import os
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from videoscope.benchmark.host_resources import HostResourceSampler, HostResourceSnapshot
from videoscope.benchmark.measurements import ProcessRecord, ProcessTreeRssSampler
from videoscope.benchmark.smoke_timeline import SmokeTimeline


@pytest.mark.parametrize("secondary_host_failure", [False, True])
def test_failure_after_rss_start_preserves_primary_and_stops_both_samplers(tmp_path, secondary_host_failure):
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    driver = Path(os.environ["TEST_NO_ML_DRIVER"])
    primary = RuntimeError("synthetic primary failure after RSS startup")
    host_provider = SimpleNamespace(identity="fake-native-host@1", scope="system_wide", snapshot=lambda: HostResourceSnapshot(1, 2, 0, 10, 20, 16384))
    process_provider = SimpleNamespace(identity="fake-native-process@1", snapshot=lambda: (ProcessRecord(os.getpid(), 1, 123, "owner", "python"),))
    host_finishes = []
    rss_starts = []
    rss_closes = []
    original_host_finish = HostResourceSampler.finish
    original_rss_start = ProcessTreeRssSampler.start
    original_rss_close = ProcessTreeRssSampler.close

    def host_finish(sampler):
        host_finishes.append(sampler)
        result = original_host_finish(sampler)
        if secondary_host_failure:
            raise RuntimeError("synthetic secondary cleanup failure")
        return result

    def rss_start(sampler):
        rss_starts.append(sampler)
        return original_rss_start(sampler)

    def rss_close(sampler):
        rss_closes.append(sampler)
        return original_rss_close(sampler)

    def fail_event(_timeline, event_id):
        assert event_id == "idle.begin"
        assert len(rss_starts) == 1 and rss_starts[0]._started
        raise primary

    def command(args, **_kwargs):
        if args[0] == "/bin/ps":
            return ""
        if "rev-parse" in args:
            return "7054f137f92c18676461242a18c8fc2619898b68\n"
        return b""

    with patch("sys.argv", [str(driver), str(tmp_path), str(out)]), patch("subprocess.check_output", side_effect=command), patch("videoscope.benchmark.host_resources.create_host_resource_snapshot_provider", return_value=host_provider), patch("videoscope.benchmark.measurements.create_native_process_snapshot_provider", return_value=process_provider), patch.object(HostResourceSampler, "finish", host_finish), patch.object(ProcessTreeRssSampler, "start", rss_start), patch.object(ProcessTreeRssSampler, "close", rss_close), patch.object(SmokeTimeline, "record_event", fail_event):
        with pytest.raises(RuntimeError) as caught:
            runpy.run_path(str(driver), run_name="__main__")

    assert caught.value is primary
    assert len(host_finishes) == len(rss_closes) == 1
    assert rss_closes == rss_starts
    assert not host_finishes[0]._thread.is_alive()
    assert not rss_closes[0]._thread.is_alive()
    assert (out / "no-ml-control.json").read_bytes() == b""

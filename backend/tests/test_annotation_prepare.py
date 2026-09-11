from pathlib import Path
import json

import pytest

from videoscope.annotation_review.prepare import (
    clip_command, parse_sample_times, select_windows, selected_sources,
)


def test_selection_mixes_uniform_controls_and_diverse_proposals_deterministically():
    times = list(range(0, 1000, 4))
    scores = [[float((i + k * 37) % 97) for k in range(5)] for i in range(len(times))]
    result = select_windows(times, scores, 1000.0)
    assert result == select_windows(times, scores, 1000.0)
    assert len(result) == 12
    assert sum(r['selection_method'] == 'timeline_uniform' for r in result) == 4
    assert all(0 <= r['start'] < r['end'] <= 1000 for r in result)
    assert all(abs(a['center']-b['center']) >= 30 for i,a in enumerate(result) for b in result[i+1:])


@pytest.mark.parametrize('times,scores,duration', [
    ([1], [[float('nan')]*5], 100),
    ([1, 0], [[0]*5]*2, 100),
    ([1], [[0]*4], 100),
    ([1], [[0]*5], float('inf')),
    ([101], [[0]*5], 100),
])
def test_selection_rejects_invalid_measurements(times, scores, duration):
    with pytest.raises(ValueError):
        select_windows(times, scores, duration)


def test_sparse_short_source_is_honest_not_duplicated_to_fill_quota():
    rows = select_windows([0, 4, 8], [[0]*5]*3, 12)
    assert len(rows) == 1
    assert rows[0]['start'] == 0 and rows[0]['end'] == 12


def test_actual_frame_pts_are_parsed_and_discontinuities_rejected():
    log = '[Parsed_showinfo] n:   0 pts: 0 pts_time:0 duration:1\n[Parsed_showinfo] n: 1 pts: 65536 pts_time:4.26667 duration:1\n'
    assert parse_sample_times(log) == [0.0, 4.26667]
    with pytest.raises(ValueError):
        parse_sample_times(log + '[Parsed_showinfo] n: 2 pts: 0 pts_time:0 duration:1\n')
    with pytest.raises(ValueError):
        parse_sample_times('nothing decoded')


def test_source_roles_are_opt_in_and_reserved_media_never_resolved(tmp_path):
    source = tmp_path / 'source.mp4'; source.write_bytes(b'video')
    inventory = {'sources':[{'source_id':'youtube:a','source_relpath':'source.mp4','bytes':5}, {'source_id':'youtube:b','source_relpath':'absent.mp4','bytes':3}]}
    roles = {'sources':[{'source_id':'youtube:a','role':'development_review','content_decode_allowed_current_slice':True,'training_allowed':False,'promotion_eligible':False}, {'source_id':'youtube:b','role':'reserve_uninspected','content_decode_allowed_current_slice':False}]}
    selected = selected_sources(inventory, roles, tmp_path)
    assert len(selected) == 1 and selected[0]['path'] == source
    source.unlink(); source.symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        selected_sources(inventory, roles, tmp_path)


def test_source_path_escape_and_role_drift_fail_closed(tmp_path):
    inventory = {'sources':[{'source_id':'a','source_relpath':'../secret','bytes':2}]}
    roles = {'sources':[{'source_id':'a','role':'development_review','content_decode_allowed_current_slice':True,'training_allowed':False,'promotion_eligible':False}]}
    with pytest.raises(ValueError):
        selected_sources(inventory, roles, tmp_path)
    roles['sources'][0]['role'] = 'promotion_holdout'
    with pytest.raises(ValueError):
        selected_sources(inventory, roles, tmp_path)


def test_clip_export_is_bounded_native_rate_and_never_overwrites():
    cmd = clip_command('/ffmpeg', Path('/source name.mp4'), Path('/new.mp4'), 8.5, 18)
    assert '-n' in cmd and '-y' not in cmd
    assert '-r' not in cmd
    assert cmd[cmd.index('-i')+1] == '/source name.mp4'
    assert cmd[cmd.index('-t')+1] == '18.000000'
    with pytest.raises(ValueError):
        clip_command('/ffmpeg', Path('/a'), Path('/a'), 0, 18)
    with pytest.raises(ValueError):
        clip_command('/ffmpeg', Path('/a'), Path('/b'), -1, 18)

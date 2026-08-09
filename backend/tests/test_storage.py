from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from videoscope.storage import atomic_write_json


def test_atomic_json_writes_use_independent_temporary_files(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    payloads = [{"writer": index, "value": "x" * 4_096} for index in range(12)]

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(lambda payload: atomic_write_json(destination, payload), payloads))

    assert json.loads(destination.read_text(encoding="utf-8")) in payloads
    assert list(tmp_path.glob(".state.json.*.tmp")) == []

#!/bin/zsh
# Open the prepared private workbench on this Mac, including after a restart.
set -eu
project_root="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$project_root"
export PYTHONPATH="$project_root/backend/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$project_root/.venv/bin/python" - "$project_root" <<'PY'
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

root = Path(sys.argv[1])
workspace = root / 'data/ml/reviews/uba-pilot-v1/workbench-v1'
manifest = workspace / 'workspace.json'
origin = 'http://127.0.0.1:8767'
if not manifest.is_file():
    raise SystemExit('Рабочее место ещё не подготовлено. Инструкция: docs/benchmarks/phase1/workbench/README.md')
revision = json.loads(manifest.read_text())['workspace_revision']
local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def ready():
    try:
        with local_http.open(origin + '/api/workspace', timeout=1) as response:
            result = json.load(response)
    except urllib.error.URLError:
        return False
    if result.get('workspace_revision') != revision:
        raise SystemExit('Порт 8767 занят другим рабочим местом. Существующий сервер оставлен без изменений.')
    return True

if ready():
    webbrowser.open(origin)
    raise SystemExit(0)

server = subprocess.Popen([sys.executable, '-m', 'videoscope.annotation_workbench.server',
                           '--workspace', str(workspace), '--port', '8767'])
try:
    for _ in range(80):
        if server.poll() is not None:
            raise SystemExit('Сервер не запустился; описание ошибки находится выше.')
        if ready():
            print('Разметка открыта. Оставьте это окно работающим; сохранённые ответы останутся после его закрытия.')
            webbrowser.open(origin)
            server.wait()
            break
        time.sleep(.25)
    else:
        raise SystemExit('Сервер не ответил вовремя. Сохранённые данные не изменены.')
except KeyboardInterrupt:
    pass
finally:
    if server.poll() is None:
        server.terminate()
        server.wait(timeout=10)
PY

import hashlib, json, os, re, shlex, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CHECKOUT = ROOT / 'checkout'
PRIVATE = ROOT / 'private'
SHA = '7054f137f92c18676461242a18c8fc2619898b68'
OLD_BOOT = 1787056703

def write_new(name, payload):
    with (PRIVATE / name).open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write('\n')

def clean_sha():
    actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
    assert actual == SHA
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT)
    return actual

def workers():
    rows = subprocess.check_output(['/bin/ps','-ww','-axo','pid=,ppid=,args='],text=True)
    found = []
    for row in rows.splitlines():
        parts = row.split(None,2)
        if len(parts)!=3: continue
        try: args=shlex.split(parts[2])
        except ValueError: continue
        if any(arg=='-m' and i+1<len(args) and args[i+1].startswith(('videoscope.providers.','videoscope.benchmark.')) for i,arg in enumerate(args)):
            found.append({'pid':int(parts[0]),'ppid':int(parts[1])})
    return found

boot_raw = subprocess.check_output(['/usr/sbin/sysctl','kern.boottime'],text=True)
boot = int(re.search(r'sec = (\d+)',boot_raw).group(1))
assert boot > OLD_BOOT
clean_sha()
assert not workers()
smoke_root = ROOT / 'smoke-root'
assert smoke_root.is_dir() and not smoke_root.is_symlink()
assert smoke_root.stat().st_mode & 0o777 == 0o700
assert not list(smoke_root.iterdir())
attestation_raw = (PRIVATE/'ml-environment.json').read_bytes()
attestation = json.loads(attestation_raw)
assert attestation['status']=='complete' and not attestation['failures']
for directory in ('.venv','.venv-vision','.venv-whisper','.venv-ocr','.venv-lighthouse','.venv-qwen'):
    assert (CHECKOUT/directory/'bin/python').is_file()
if '--preflight-only' in sys.argv:
    print(json.dumps({'status':'ready','code_sha':SHA,'boot_unix_seconds':boot,'other_ml_workers':0,'environments':'attested','smoke_attempt_started':False}))
    raise SystemExit(0)
plan = {
'schema_version':1,'code_sha':SHA,'declared_at_utc':datetime.now(timezone.utc).isoformat(),
'control_kind':'normal_full_smoke_after_owner_fresh_boot','control_command':'make full-ml-smoke',
'previous_boot_unix_seconds':OLD_BOOT,'boot_unix_seconds':boot,
'hypothesis':'An unchanged normal full smoke in a fresh owner-provided macOS session may satisfy the existing zero-swap gate. Passing does not establish per-process attribution or production quality.',
'maximum_attempts':1,'principal_change':'host_session_only','coverage_changed':False,'runtime_changed':False,
'zero_swap_gate_changed':False,'background_counts_subtracted':False,'training':False,'external_data_import':False,'later_phases':False,
'fresh_offline_environments':6,'environment_attestation_sha256':hashlib.sha256(attestation_raw).hexdigest(),
'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
'agent_helper_process_lease':'No parallel tools, readers, tests or scans; root polls existing launcher only.',
'preflight_matching_ml_workers':0,'declared_before_inference':True,
}
write_new('fresh-boot-control-plan.json',plan)
user_home = str(Path.home())
env = {'HOME':user_home,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
'HF_HOME':user_home+'/.cache/huggingface','HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
'TRANSFORMERS_OFFLINE':'1','UV_OFFLINE':'1','UV_PYTHON_DOWNLOADS':'never','PYTHONNOUSERSITE':'1',
'PYTHONDONTWRITEBYTECODE':'1','PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK':'True',
'VIDEOSCOPE_OCR_MODEL_ROOT':user_home+'/.paddlex/official_models',
'VIDEOSCOPE_FULL_ML_SMOKE_ROOT':str(smoke_root),'VIDEOSCOPE_FULL_ML_SMOKE_MODELS_ROOT':str(ROOT/'models'),
'VIDEOSCOPE_FULL_ML_SMOKE_FFMPEG_BINARY':'/opt/homebrew/bin/ffmpeg',
'VIDEOSCOPE_FULL_ML_SMOKE_FFPROBE_BINARY':'/opt/homebrew/bin/ffprobe'}
for role in ('vision','whisper','ocr','lighthouse','qwen'):
    env['VIDEOSCOPE_FULL_ML_SMOKE_'+role.upper()+'_PYTHON']=str(CHECKOUT/('.venv-'+role)/'bin/python')
started=time.monotonic()
with (PRIVATE/'full-ml-smoke-fresh-boot.json').open('xb') as out, (PRIVATE/'full-ml-smoke-fresh-boot.stderr.log').open('xb') as err:
    result=subprocess.run(['make','full-ml-smoke'],cwd=CHECKOUT,env=env,stdout=out,stderr=err,timeout=1200)
elapsed=time.monotonic()-started
raw=(PRIVATE/'full-ml-smoke-fresh-boot.json').read_bytes()
try: receipt=json.loads(raw)
except (ValueError,UnicodeError): receipt={}
summary={'schema_version':1,'code_sha_before':SHA,'code_sha_after':clean_sha(),'boot_unix_seconds':boot,
'command_exit_code':result.returncode,'elapsed_seconds':elapsed,'stdout_bytes':len(raw),'stdout_sha256':hashlib.sha256(raw).hexdigest(),
'smoke_status':receipt.get('status'),'workspace_cleanup':receipt.get('workspace_cleanup'),
'smoke_root_empty':not list(smoke_root.iterdir()),'postflight_matching_ml_workers':workers()}
write_new('fresh-boot-control-launch-result.json',summary)
print(json.dumps(summary),flush=True)
raise SystemExit(result.returncode)

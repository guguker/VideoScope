from pathlib import Path
from datetime import datetime,timezone
import dataclasses,hashlib,json,os,shlex,subprocess,sys,time,threading
from videoscope.benchmark.host_resources import HostResourceSampler,create_host_resource_snapshot_provider
from videoscope.benchmark.measurements import ProcessTreeRssSampler,create_native_process_snapshot_provider
from videoscope.benchmark.smoke_timeline import SmokeTimeline
CHECKOUT=Path(sys.argv[1]).resolve(); OUTPUT=Path(sys.argv[2]).resolve()
assert OUTPUT.is_dir() and OUTPUT.stat().st_mode&0o777==0o700
SHA=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
assert SHA=='7054f137f92c18676461242a18c8fc2619898b68'
assert subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT)==b''
def workers_count():
    found=[]
    for line in subprocess.check_output(['/bin/ps','-ww','-axo','pid=,ppid=,args='],text=True).splitlines():
        fields=line.split(None,2)
        if len(fields)!=3: continue
        try: args=shlex.split(fields[2])
        except ValueError: continue
        if any(arg=='-m' and args[i+1].startswith(('videoscope.providers.','videoscope.benchmark.')) for i,arg in enumerate(args[:-1])):found.append(int(fields[0]))
    return len(found)
assert workers_count()==0
plan={'schema_version':1,'code_sha':SHA,'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'declared_at_utc':datetime.now(timezone.utc).isoformat(),'declared_before_measurement':True,'control_kind':'no_ml_sampler_control_corrected_driver','previous_control_status':'invalid_no_raw_receipt','correction':'Use actual HostResourceSampler.finish lifecycle; preserve primary error and always stop both samplers' ,'maximum_attempts':1,'duration_after_baseline_seconds':120,'managed_workers':0,'inference_calls':0,'process_sample_interval_milliseconds':50,'host_sample_interval_milliseconds':250,'measurement_use':'diagnostic_only_not_gate_evidence','promotion_eligible':False,'hypothesis':'System-wide swap counters may advance while only existing native samplers and their timeline observer run, without any VideoScope model inference. Positive events do not identify their causing process; no counts may be subtracted from a full smoke.','host_cold_state':'not_established','helper_lease':'Root polls existing launcher only; no agent tools or tests during the measurement.'}
with (OUTPUT/'no-ml-control-plan.json').open('x') as f:json.dump(plan,f,sort_keys=True,indent=2);f.write('\n')
# Reserve destinations before sampling, never overwrite prior diagnostics.
with (OUTPUT/'no-ml-control.json').open('x') as out:
    trace=SmokeTimeline(frozenset({'idle.begin','idle.end'}))
    host=HostResourceSampler(create_host_resource_snapshot_provider())
    rss=ProcessTreeRssSampler(root_pid=os.getpid(),provider=create_native_process_snapshot_provider(root_pid=os.getpid()),diagnostic_observer=trace.observe_process_snapshot)
    host_started=False;failure=None;resources=None
    try:
        host.start();host_started=True;trace.record_host_sampler_epoch(host.started_monotonic_ns);rss.start()
        trace.record_event('idle.begin');start=time.monotonic();deadline=start+120.0
        while time.monotonic()<deadline:threading.Event().wait(min(30.0,deadline-time.monotonic()))
        trace.record_event('idle.end');process=rss.finish()
    except BaseException as error:
        failure=error
    finally:
        try:rss.close()
        except BaseException as error:failure=failure or error
        if host_started:
            try:resources=host.finish()
            except BaseException as error:failure=failure or error
    if failure is not None:raise failure
    assert resources is not None
    elapsed=time.monotonic()-start
    result={'schema_version':1,'status':'complete','code_sha':SHA,'driver_sha256':plan['driver_sha256'],'measurement_use':'diagnostic_only_not_gate_evidence','promotion_eligible':False,'inference_calls':0,'managed_workers':0,'worker_count_before':0,'worker_count_after':workers_count(),'elapsed_after_baseline_seconds':elapsed,'process_tree':dataclasses.asdict(process),'host_resources':resources.to_portable_dict(),'timeline':trace.to_portable_dict(code_sha=SHA),'samplers_closed':True}
    assert subprocess.check_output(['git','status','--porcelain'],cwd=CHECKOUT)==b''
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()==SHA
    json.dump(result,out,sort_keys=True,separators=(',',':'),allow_nan=False);out.write('\n')
raw=(OUTPUT/'no-ml-control.json').read_bytes()
print(json.dumps({'status':'complete','raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_bytes':len(raw),'worker_count_after':result['worker_count_after'],'elapsed_after_baseline_seconds':elapsed,'sample_count':process.sample_count,'rss_peak_bytes':process.peak_bytes,'virtual_memory':result['host_resources']['virtual_memory'],'metal_recovery_delta':resources.metal_recovery_delta},sort_keys=True),flush=True)

import itertools,json,os,runpy,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from videoscope.benchmark.host_resources import HostResourceSnapshot
from videoscope.benchmark.measurements import ProcessRecord

def test_driver_closes_real_samplers_and_publishes(tmp_path):
    out=tmp_path/'out';out.mkdir(mode=0o700)
    driver=Path(os.environ['TEST_NO_ML_DRIVER'])
    host=SimpleNamespace(identity='fake-native-host@1',scope='system_wide',snapshot=lambda:HostResourceSnapshot(1,2,0,10,20,16384))
    process=SimpleNamespace(identity='fake-native-process@1',snapshot=lambda:(ProcessRecord(os.getpid(),1,123,'owner','python'),))
    def command(args,**kwargs):
        if args[0]=='/bin/ps':return ''
        if 'rev-parse' in args:return '7054f137f92c18676461242a18c8fc2619898b68\n'
        return b''
    with patch('sys.argv',[str(driver),str(tmp_path),str(out)]),patch('subprocess.check_output',side_effect=command),patch('time.monotonic',side_effect=itertools.count(0,121)),patch('videoscope.benchmark.host_resources.create_host_resource_snapshot_provider',return_value=host),patch('videoscope.benchmark.measurements.create_native_process_snapshot_provider',return_value=process):
        runpy.run_path(str(driver),run_name='__main__')
    data=json.loads((out/'no-ml-control.json').read_bytes())
    assert data['samplers_closed'] is True
    assert data['status']=='complete'
    assert data['host_resources']['virtual_memory']['swapins_delta_pages']==0
    assert data['worker_count_after']==0
    assert [row['total_rss_bytes'] for row in data['timeline']['process_samples']]==data['process_tree']['samples_bytes']

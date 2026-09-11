import base64
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from test_annotation_workbench_storage import workspace_fixture, save, shot


def add_legacy(legacy,source_sha):
    from videoscope.annotation_review.schema import canonical_json,batch_revision
    batch=json.loads((legacy/'batch.json').read_text())
    batch['examples']=[dict(example_id='clip-a',source_id='alias-a',source_start_seconds=3600.,source_end_seconds=3618.,clip_duration_seconds=18.,prepared_input_sha256='c'*64)]
    (legacy/'batch.json').write_bytes(canonical_json(batch))
    annotation=dict(schema_version=1,batch_id='pilot',batch_revision=batch_revision(batch),example_id='clip-a',revision=1,created_at='2026-09-11T10:00:00+00:00',reviewer='local_owner',label_status='human_reviewed',destination='annotation_inbox',gold=False,training_allowed=False,promotion_allowed=False,source_id='alias-a',source_sha256=source_sha,source_start_seconds=3600.,source_end_seconds=3618.,prepared_input_sha256='c'*64,shot_type='two',outcome='made',presentation='live',boundary_status='complete',start_seconds=2.,end_seconds=7.,notes='old answer')
    raw=json.dumps(annotation,indent=3).encode()+b'\n'
    directory=legacy/'annotations'/'clip-a';directory.mkdir(parents=True)
    (directory/'000001.json').write_bytes(raw)
    return raw


def test_legacy_exact_bytes_alias_conversion_and_old_completion(tmp_path):
    store,source,legacy=workspace_fixture(tmp_path)
    raw=add_legacy(legacy,hashlib.sha256(source.read_bytes()).hexdigest())
    from videoscope.annotation_workbench.legacy import import_legacy
    assert import_legacy(store,legacy)['legacy_revisions_added']==1
    record=store.source('uba-01')['records'][0]
    assert record['data']['start_seconds']==3602
    assert record['data']['end_seconds']==3607
    assert record['status']=='human_reviewed'
    assert record['data']['scoring_decision'] is None
    assert record['legacy']['label_status']=='human_reviewed'
    assert (legacy/'annotations/clip-a/000001.json').read_bytes()==raw
    backup=[json.loads(line) for line in store.export_lines()]
    assert base64.b64decode(next(line for line in backup if line['type']=='legacy')['raw_base64'])==raw
    assert import_legacy(store,legacy)['legacy_revisions_added']==0


def test_portable_restore_into_new_initialized_root_and_tail(tmp_path):
    from videoscope.annotation_workbench.prepare import prepare_workspace
    from videoscope.annotation_workbench.storage import WorkbenchStore
    store,_,legacy=workspace_fixture(tmp_path)
    first=save(store,data=shot())
    backup=list(store.export_lines())
    other=tmp_path/'different-root'
    prepare_workspace(audit_dir=tmp_path/'audit',project_root=tmp_path,legacy_batch=legacy,output=other,code_sha='b'*40)
    target=WorkbenchStore(other)
    assert target.revision==store.revision
    assert target.restore_lines(backup)['added']==1
    second=save(store,data={**shot(),'notes':'new'},record_id=first['record_id'],expected_revision=1)
    assert target.restore_lines(store.export_lines())['added']==1
    assert target.history(first['record_id'])[-1]==second
    # Divergent suffix, with a valid trailer, must not partially insert another entity.
    save(target,data={**shot(),'notes':'different'},record_id=first['record_id'],expected_revision=2)
    save(store,data={**shot(),'notes':'original'},record_id=first['record_id'],expected_revision=2)
    save(store,data=shot())
    before=list(target.export_lines())
    with pytest.raises(ValueError): target.restore_lines(store.export_lines())
    assert list(target.export_lines())==before


def test_prepare_rejects_existing_output_and_rights_hash_mismatch(tmp_path):
    from videoscope.annotation_workbench.prepare import prepare_workspace
    store,_,legacy=workspace_fixture(tmp_path)
    with pytest.raises(ValueError): prepare_workspace(audit_dir=tmp_path/'audit',project_root=tmp_path,legacy_batch=legacy,output=store.root,code_sha='a'*40)
    p=tmp_path/'audit/source-rights-ledger.json';data=json.loads(p.read_text());data['sources'][0]['source_sha256']='f'*64;p.write_text(json.dumps(data))
    with pytest.raises(ValueError): prepare_workspace(audit_dir=tmp_path/'audit',project_root=tmp_path,legacy_batch=legacy,output=tmp_path/'bad',code_sha='a'*40)
    assert not (tmp_path/'bad').exists()


def test_legacy_v1_requires_current_review_without_changing_original_status(tmp_path):
    store,source,legacy=workspace_fixture(tmp_path)
    add_legacy(legacy,hashlib.sha256(source.read_bytes()).hexdigest())
    from videoscope.annotation_workbench.legacy import import_legacy
    import_legacy(store,legacy)
    record=store.source('uba-01')['records'][0]
    assert record['needs_review'] is True
    assert record['legacy']['label_status']=='human_reviewed'


def test_public_fps_comes_from_inventory_not_media_probe(tmp_path):
    from videoscope.annotation_workbench.prepare import prepare_workspace
    from videoscope.annotation_workbench.storage import WorkbenchStore
    _,_,legacy=workspace_fixture(tmp_path)
    p=tmp_path/'audit/inventory.json';value=json.loads(p.read_text())
    value['sources'][0]['video_streams']=[{'avg_frame_rate':'60000/1001'}]
    value['sources'][1]['video_streams']=[{'avg_frame_rate':'60/1'}]
    p.write_text(json.dumps(value))
    root=tmp_path/'fps-root'
    prepare_workspace(audit_dir=p.parent,project_root=tmp_path,legacy_batch=legacy,output=root,code_sha='a'*40)
    sources=WorkbenchStore(root).workspace()['sources']
    assert sources[0]['fps']==pytest.approx(60000/1001)
    assert sources[1]['fps']==60


def rewrite_backup(lines, mutate):
    values=[json.loads(line) for line in lines]
    mutate(values)
    raw=[json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()+b'\n' for v in values[:-1]]
    values[-1]['sha256']=hashlib.sha256(b''.join(raw)).hexdigest()
    return raw+[json.dumps(values[-1],sort_keys=True,separators=(',',':')).encode()+b'\n']


def test_restore_rejects_forged_legacy_metadata_and_orphan_raw_atomically(tmp_path):
    store,source,legacy=workspace_fixture(tmp_path)
    add_legacy(legacy,hashlib.sha256(source.read_bytes()).hexdigest())
    from videoscope.annotation_workbench.legacy import import_legacy
    import_legacy(store,legacy)
    backup=list(store.export_lines())
    bad=rewrite_backup(backup,lambda rows:next(r for r in rows if r['type']=='record')['record']['legacy'].update(path='/private/user/video'))
    with pytest.raises(ValueError): store.restore_lines(bad)
    def orphan(rows):
        rows[:]=[r for r in rows if r['type']!='record']
        rows[-1]['counts']['records']=0
    bad=rewrite_backup(backup,orphan)
    with pytest.raises(ValueError):store.restore_lines(bad)
    assert list(store.export_lines())==backup


def test_restore_retains_progress_and_drafts_in_new_workspace(tmp_path):
    from videoscope.annotation_workbench.prepare import prepare_workspace
    from videoscope.annotation_workbench.storage import WorkbenchStore
    store,_,legacy=workspace_fixture(tmp_path)
    draft=save(store,data={'start_seconds':3600.,'notes':'unfinished'})
    store.progress(dict(schema_version=1,workspace_revision=store.revision,source_id='uba-01',position_seconds=3600.,selected_record_id=draft['record_id'],playback_rate=1.25))
    prepare_workspace(audit_dir=tmp_path/'audit',project_root=tmp_path,legacy_batch=legacy,output=tmp_path/'fresh',code_sha='x')
    target=WorkbenchStore(tmp_path/'fresh')
    target.restore_lines(store.export_lines())
    assert target.source('uba-01')==store.source('uba-01')


def test_integer_audit_duration_uses_one_canonical_public_identity(tmp_path):
    from videoscope.annotation_workbench.prepare import prepare_workspace
    from videoscope.annotation_workbench.storage import WorkbenchStore
    _,_,legacy=workspace_fixture(tmp_path)
    p=tmp_path/'audit/inventory.json';value=json.loads(p.read_text())
    for source in value['sources']:source['duration_seconds']=7200
    p.write_text(json.dumps(value))
    root=tmp_path/'integer-duration'
    result=prepare_workspace(audit_dir=p.parent,project_root=tmp_path,legacy_batch=legacy,output=root,code_sha='a'*40)
    assert WorkbenchStore(root).revision==result['workspace_revision']


def test_cross_source_player_reference_is_rejected(tmp_path):
    from videoscope.annotation_workbench.prepare import prepare_workspace
    from videoscope.annotation_workbench.storage import WorkbenchStore
    _,_,legacy=workspace_fixture(tmp_path)
    p=tmp_path/'audit/inventory.json';value=json.loads(p.read_text())
    second=tmp_path/'second.mp4';second.write_bytes(b'second source')
    value['sources'][1].update(source_sha256=hashlib.sha256(second.read_bytes()).hexdigest(),source_relpath=second.name,bytes=second.stat().st_size)
    p.write_text(json.dumps(value))
    roles=tmp_path/'audit/source-roles.json';rows=json.loads(roles.read_text());rows['sources'][1].update(role='development_review',content_decode_allowed_current_slice=True);roles.write_text(json.dumps(rows))
    rights=tmp_path/'audit/source-rights-ledger.json';rows=json.loads(rights.read_text());rows['sources'][1]['source_sha256']=value['sources'][1]['source_sha256'];rights.write_text(json.dumps(rows))
    root=tmp_path/'two-sources'
    prepare_workspace(audit_dir=p.parent,project_root=tmp_path,legacy_batch=legacy,output=root,code_sha='a'*40)
    store=WorkbenchStore(root)
    player=save(store,'player',{'number_status':'unreadable'},source_id='uba-02',status='human_reviewed')
    with pytest.raises(ValueError):save(store,data={**shot(),'actor_id':player['record_id']})

import hashlib
import json
import sqlite3
from uuid import uuid4

import pytest


def workspace_fixture(tmp_path):
    from videoscope.annotation_workbench.prepare import prepare_workspace
    audit = tmp_path / 'audit'; audit.mkdir()
    source = tmp_path / 'source.mp4'; source.write_bytes(bytes(range(256)))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    entries = [dict(source_id='original-a', source_sha256=digest, source_relpath='source.mp4', bytes=256, duration_seconds=7200., grouping={'source_group_id':'game-a'}), dict(source_id='original-b',source_sha256='b'*64,source_relpath='DO-NOT-READ.mp4',bytes=256,duration_seconds=7200., grouping={'source_group_id':'game-b'})]
    (audit/'inventory.json').write_text(json.dumps({'sources':entries}))
    (audit/'source-roles.json').write_text(json.dumps({'sources':[dict(source_id=e['source_id'],role='development_review' if i==0 else 'reserve_uninspected',content_decode_allowed_current_slice=i==0) for i,e in enumerate(entries)]}))
    (audit/'source-rights-ledger.json').write_text(json.dumps({'sources':[dict(source_id=e['source_id'],source_sha256=e['source_sha256'],rights_status='unknown') for e in entries]}))
    legacy = tmp_path/'legacy'; legacy.mkdir()
    # No annotations is a valid pilot state; manifest can contain zero examples for synthetic tests.
    (legacy/'batch.json').write_text(json.dumps({'schema_version':1,'batch_id':'pilot','sources':[{'source_id':'alias-a','sha256':digest,'duration_seconds':7200.}], 'examples':[]}))
    root=tmp_path/'workbench'
    prepare_workspace(audit_dir=audit, project_root=tmp_path,legacy_batch=legacy,output=root,code_sha='a'*40)
    from videoscope.annotation_workbench.storage import WorkbenchStore
    return WorkbenchStore(root), source, legacy


@pytest.fixture
def store(tmp_path):
    return workspace_fixture(tmp_path)[0]


def save(store,kind='event',data=None,**extra):
    return store.save({'schema_version':1,'workspace_revision':store.workspace()['workspace_revision'],'source_id':'uba-01','record_id':kind+'-'+uuid4().hex,'expected_revision':0,'kind':kind,'status':'draft','archived':False,'data':data or {},**extra})


def shot():
    return dict(event_type='shot',start_seconds=12.,end_seconds=17.,shot_type='two',outcome='miss',scoring_decision='not_applicable',play_context='foul_on_shot',presentation='live',boundary_status='complete')


def revise(store,record,**changes):
    return save(store,record['kind'],record['data'],record_id=record['record_id'],expected_revision=record['revision'],status=record['status'],archived=record['archived'],**changes)


def test_reviewed_shot_and_partial_draft_survive_restart(store):
    from videoscope.annotation_workbench.storage import WorkbenchStore
    reviewed=save(store,data=shot(),status='human_reviewed')
    draft=save(store,data={'start_seconds':0.,'notes':'incomplete'})
    rows=WorkbenchStore(store.root).source('uba-01')['records']
    assert {r['record_id'] for r in rows}=={reviewed['record_id'],draft['record_id']}
    assert next(r for r in rows if r['record_id']==draft['record_id'])['data']['start_seconds']==0
    assert reviewed['revision']==1


def test_stale_write_and_immutable_sql_history(store):
    from videoscope.annotation_workbench.storage import WorkbenchConflict
    first=save(store,data=shot())
    second=revise(store,first)
    with pytest.raises(WorkbenchConflict): revise(store,first)
    assert len(store.history(first['record_id']))==2
    with sqlite3.connect(store.root/'annotations.sqlite3') as db:
        with pytest.raises(sqlite3.DatabaseError): db.execute('DELETE FROM revisions')
    assert second['revision']==2


def test_player_numbers_preserve_zero_and_double_zero(store):
    for number in ('0','00'):
        player=save(store,'player',{'jersey_number':number,'number_status':'readable'},status='human_reviewed')
        assert player['data']['jersey_number']==number
    with pytest.raises(ValueError): save(store,'player',{'jersey_number':'00','number_status':'offscreen'})


def test_reference_and_reverse_invalidation(store):
    team=save(store,'team',{'name':'Blue'},status='human_reviewed')
    poss=save(store,'possession',{'start_seconds':10.,'end_seconds':20.,'team_id':team['record_id']},status='human_reviewed')
    event=save(store,data={**shot(),'possession_id':poss['record_id']},status='human_reviewed')
    with pytest.raises(ValueError):
        save(store,'possession',{**poss['data'],'end_seconds':14.},record_id=poss['record_id'],expected_revision=1,status='human_reviewed')
    with pytest.raises(ValueError):
        save(store,'team',team['data'],record_id=team['record_id'],expected_revision=1,archived=True)
    with pytest.raises(ValueError): save(store,data={**shot(),'actor_id':team['record_id']})
    with pytest.raises(ValueError): save(store,data={**shot(),'actor_id':'player-'+'f'*32})
    assert len(store.history(event['record_id']))==1


@pytest.mark.parametrize('data',[{**shot(),'play_context':'after_whistle','scoring_decision':'counted'},{**shot(),'end_seconds':None},{**shot(),'start_seconds':float('nan')},{**shot(),'unknown':'x'}])
def test_invalid_reviewed_labels_are_rejected(store,data):
    with pytest.raises(ValueError): save(store,data=data,status='human_reviewed')


def test_frame_visibility_and_duplicate_players(store):
    p=save(store,'player',{})
    pt={'point_id':uuid4().hex,'entity':'player','player_id':p['record_id'],'x':.25,'y':.75,'visibility':'visible','anchor':'floor_contact'}
    frame=save(store,'frame',{'timestamp_seconds':16.,'points':[pt]},status='human_reviewed')
    assert frame['data']['points'][0]['x']==.25
    with pytest.raises(ValueError): save(store,'frame',{'timestamp_seconds':16.,'points':[pt,{**pt,'point_id':uuid4().hex}]})
    with pytest.raises(ValueError): save(store,'frame',{'timestamp_seconds':16.,'points':[{**pt,'visibility':'offscreen'}]})


def test_backup_roundtrip_tail_idempotence_progress_and_atomic_rejection(store):
    from videoscope.annotation_workbench.storage import WorkbenchStore
    first=save(store,data=shot())
    store.progress({'schema_version':1,'workspace_revision':store.workspace()['workspace_revision'],'source_id':'uba-01','position_seconds':3666.,'selected_record_id':first['record_id'],'playback_rate':1.5})
    backup=list(store.export_lines())
    assert store.restore_lines(iter(backup))['added']==0
    second=revise(store,first)
    assert store.restore_lines(iter(backup))['added']==0
    before=list(store.export_lines())
    for bad in (backup[:-1],backup+[backup[-1]],backup[:1]+[b'{"type":"record"}\n']+backup[2:]):
        with pytest.raises(ValueError): store.restore_lines(iter(bad))
        assert list(store.export_lines())==before
    assert WorkbenchStore(store.root).workspace()['progress']['position_seconds']==3666
    assert second['revision']==2


def test_reserved_source_is_metadata_only_and_public_data_has_no_paths(store):
    w=store.workspace()
    assert w['sources'][1]['media_url'] is None
    with pytest.raises(ValueError): store.source('uba-02')
    with pytest.raises(ValueError): save(store,source_id='uba-02')
    assert str(store.root.parent) not in json.dumps(w)
    assert b'DO-NOT-READ' not in b''.join(store.export_lines())


def test_schema_version_cannot_be_boolean_and_partial_draft_keeps_boundary(store):
    with pytest.raises(ValueError): save(store,schema_version=True)
    saved=save(store,data={'start_seconds':22.})
    assert saved['data']['end_seconds'] is None


def test_reviewed_player_requires_explicit_number_observation(store):
    with pytest.raises(ValueError): save(store,'player',{},status='human_reviewed')
    saved=save(store,'player',{'number_status':'unreadable'},status='human_reviewed')
    assert saved['data']['jersey_number'] is None


def test_database_and_manifest_symlink_drift_fail_closed(store):
    import shutil
    db=store.root/'annotations.sqlite3'
    backup=store.root/'original.sqlite3'
    shutil.copyfile(db,backup)
    db.unlink();db.symlink_to(backup)
    with pytest.raises(ValueError):save(store,data=shot())
    db.unlink();shutil.copyfile(backup,db)
    manifest=store.root/'workspace.json'
    manifest.rename(store.root/'original-manifest.json')
    manifest.symlink_to(store.root/'original-manifest.json')
    with pytest.raises(ValueError):store.workspace()

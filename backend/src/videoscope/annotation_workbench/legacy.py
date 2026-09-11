"""Read-only pilot migration. Alias matching and raw bytes never infer labels."""
import base64
import hashlib
from uuid import NAMESPACE_URL, uuid5
from .source_library import canonical, parse_json, read_json, safe_path, WorkbenchConflict
from .schema import Event, LegacyContext
from videoscope.annotation_review.schema import AnnotationRecordV1, AnnotationRecord, EventAnnotationRecord

MODELS={1:AnnotationRecordV1,2:AnnotationRecord,3:EventAnnotationRecord}

def legacy_record(raw,context):
    LegacyContext.model_validate(context)
    value=parse_json(raw)
    if type(value.get('schema_version')) is not int or value['schema_version'] not in MODELS: raise ValueError('invalid legacy schema version')
    old=MODELS[value['schema_version']].model_validate(value)
    event_id=value.get('event_id','primary')
    if (value['batch_revision']!=context['batch_revision'] or value['example_id']!=context['example_id'] or event_id!=context['event_id'] or value['source_sha256']!=context['source_sha256'] or value['source_start_seconds']!=context['source_start_seconds'] or value['source_end_seconds']!=context['source_end_seconds']):
        raise ValueError('legacy provenance mismatch')
    data={k:value.get(k) for k in ('shot_type','outcome','scoring_decision','play_context','presentation','boundary_status','start_seconds','end_seconds','notes')}
    for key in ('start_seconds','end_seconds'):
        if data[key] is not None:
            if data[key]>context['source_end_seconds']-context['source_start_seconds']+.5: raise ValueError('legacy time exceeds clip')
            data[key]+=context['source_start_seconds']
    identity=':'.join([context['batch_revision'],context['example_id'],event_id])
    rid='event-'+uuid5(NAMESPACE_URL,identity).hex
    return dict(record_id=rid,revision=old.revision,source_id=context['source_id'],source_sha256=old.source_sha256,kind='event',status=old.label_status,archived=False,data=Event.model_validate(data).model_dump(),created_at=old.created_at,updated_at=old.created_at,origin='legacy',needs_review=old.label_status!='human_reviewed' or old.schema_version==1,legacy={**context,'schema_version':old.schema_version,'label_status':old.label_status})


def import_legacy(store,batch_dir):
    batch_dir=safe_path(batch_dir)
    batch=read_json(batch_dir/'batch.json')
    revision=hashlib.sha256(canonical(batch)).hexdigest()
    mapping={}
    for source in batch['sources']:
        matches=[s for s in store.sources.values() if s['sha256']==source['sha256'] and abs(s['duration_seconds']-source['duration_seconds'])<=.5]
        if len(matches)!=1 or not matches[0]['review_allowed']: raise ValueError('legacy source alias has no authorized exact hash/duration match')
        mapping[source['source_id']]=matches[0]['source_id']
    examples={e['example_id']:e for e in batch['examples']}
    annotations=batch_dir/'annotations'
    if not annotations.exists(): return {'legacy_revisions_added':0}
    safe_path(annotations)
    added=0
    with store._db(write=True) as db:
        for example_dir in sorted(annotations.iterdir()):
            if example_dir.name=='.lock': continue
            if example_dir.name not in examples or not example_dir.is_dir(): raise ValueError('unknown legacy example directory')
            safe_path(example_dir)
            example=examples[example_dir.name]
            paths=[('primary',example_dir)]
            extra=example_dir/'events'
            if extra.exists():
                safe_path(extra)
                paths.extend((p.name,safe_path(p)) for p in sorted(extra.iterdir()))
            for event_id,directory in paths:
                expected=1
                for path in sorted(directory.iterdir()):
                    if path.name=='events' or path.name.startswith('.pending-'): continue
                    safe_path(path)
                    if path.name!=f'{expected:06d}.json': raise ValueError('legacy revision gap')
                    with path.open('rb') as f: raw=f.read(16385)
                    if len(raw)>16384: raise ValueError('legacy record exceeds bound')
                    sid=mapping[example['source_id']]
                    context=dict(batch_revision=revision,example_id=example['example_id'],event_id=event_id,source_id=sid,source_sha256=store.sources[sid]['sha256'],source_start_seconds=example['source_start_seconds'],source_end_seconds=example['source_end_seconds'])
                    record=legacy_record(raw,context)
                    if record['revision']!=expected: raise ValueError('legacy revision mismatch')
                    if parse_json(raw)['prepared_input_sha256']!=example['prepared_input_sha256']: raise ValueError('legacy clip hash mismatch')
                    store._validate_record(record,legacy=True)
                    lid=record['record_id']+':'+str(record['revision'])
                    envelope=dict(type='legacy',legacy_id=lid,context=context,sha256=hashlib.sha256(raw).hexdigest(),raw_base64=base64.b64encode(raw).decode())
                    existing=db.execute('SELECT body FROM legacy WHERE legacy_id=?',(lid,)).fetchone()
                    if existing:
                        if existing[0]!=canonical(envelope): raise WorkbenchConflict('legacy bytes diverged')
                    else:
                        prior=db.execute('SELECT body FROM revisions WHERE record_id=? AND revision=?',(record['record_id'],record['revision'])).fetchone()
                        if prior: raise WorkbenchConflict('legacy revision conflicts with existing annotation')
                        db.execute('INSERT INTO legacy VALUES(?,?)',(lid,canonical(envelope)))
                        first=db.execute('SELECT body FROM revisions WHERE record_id=? ORDER BY revision LIMIT 1',(record['record_id'],)).fetchone()
                        if first: record['created_at']=parse_json(first[0])['created_at']
                        db.execute('INSERT INTO revisions(record_id,revision,source_id,body) VALUES(?,?,?,?)',(record['record_id'],record['revision'],sid,canonical(record)))
                        added+=1
                    expected+=1
        store._validate_references(db)
    return {'legacy_revisions_added':added}

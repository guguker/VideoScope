"""Bounded, hashed NDJSON backups staged to disk before one atomic restore."""
import base64
import hashlib
import sqlite3
import tempfile
from pydantic import TypeAdapter
from datetime import datetime
from .source_library import canonical, parse_json, WorkbenchConflict
from .schema import ProgressRequest, StoredRecord, BackupLine
from .legacy import legacy_record

BACKUP_LINE = TypeAdapter(BackupLine)

MAX_LINE_BYTES=128*1024
MAX_BACKUP_BYTES=256*1024*1024
MAX_BACKUP_LINES=1000000


def export_lines(store):
    store.check_manifest()
    digest=hashlib.sha256(); counts={'records':0,'legacy':0,'progress':0}
    def line(value):
        raw=canonical(value)+b'\n';digest.update(raw);return raw
    with store._db() as db:
        db.execute('BEGIN')
        yield line({'type':'header','format':'videoscope-workbench','schema_version':1,'workspace_revision':store.revision,'workspace':store.public})
        for row in db.execute('SELECT body FROM legacy ORDER BY legacy_id'):
            counts['legacy']+=1;yield line(parse_json(row[0]))
        for row in db.execute('SELECT body FROM revisions ORDER BY seq'):
            counts['records']+=1;yield line({'type':'record','record':parse_json(row[0])})
        for row in db.execute('SELECT body FROM progress ORDER BY source_id'):
            counts['progress']+=1;yield line({'type':'progress','progress':parse_json(row[0])})
        yield canonical({'type':'trailer','counts':counts,'sha256':digest.hexdigest()})+b'\n'


def _validate_record(store,record):
    from .storage import REQUEST
    StoredRecord.model_validate(record)
    required={'needs_review','record_id','revision','source_id','source_sha256','kind','status','archived','data','created_at','updated_at','origin'}
    if set(record)-required-{'legacy'} or required-set(record): raise ValueError('invalid exported record fields')
    if record['origin'] not in ('human','legacy'): raise ValueError('invalid record origin')
    if type(record['revision']) is not int or record['revision']<1: raise ValueError('invalid revision')
    for key in ('created_at','updated_at'):
        if len(record[key])>40 or datetime.fromisoformat(record[key]).tzinfo is None: raise ValueError('invalid timestamp')
    REQUEST.validate_python(dict(schema_version=1,workspace_revision=store.revision,source_id=record['source_id'],record_id=record['record_id'],expected_revision=record['revision']-1,kind=record['kind'],status=record['status'],archived=record['archived'],data=record['data']))
    store._validate_record(record,legacy=record['origin']=='legacy')
    needs_review=record['status']!='human_reviewed' or (record['origin']=='legacy' and record.get('legacy',{}).get('schema_version')==1)
    if record['needs_review'] is not needs_review: raise ValueError('review state mismatch')


def restore_lines(store,lines):
    store.check_manifest()
    digest=hashlib.sha256();counts={'records':0,'legacy':0,'progress':0};total=0;trailer=False
    with tempfile.TemporaryDirectory(prefix='.restore-',dir=store.root) as directory:
        stage=sqlite3.connect(directory+'/staging.sqlite3')
        try:
            stage.executescript('CREATE TABLE records(seq INTEGER PRIMARY KEY,record_id TEXT,revision INTEGER,body BLOB,UNIQUE(record_id,revision));CREATE TABLE legacy(legacy_id TEXT PRIMARY KEY,body BLOB);CREATE TABLE progress(source_id TEXT PRIMARY KEY,body BLOB);')
            for index,raw in enumerate(lines):
                if isinstance(raw,str):raw=raw.encode()
                if not isinstance(raw,bytes) or not raw.endswith(b'\n') or len(raw)>MAX_LINE_BYTES: raise ValueError('invalid backup line')
                total+=len(raw)
                if total>MAX_BACKUP_BYTES or index>=MAX_BACKUP_LINES: raise ValueError('backup exceeds bound')
                if trailer: raise ValueError('data after backup trailer')
                item=parse_json(raw)
                if not isinstance(item,dict): raise ValueError('backup line must be object')
                BACKUP_LINE.validate_python(item)
                kind=item.get('type')
                if index==0:
                    expected={'type':'header','format':'videoscope-workbench','schema_version':1,'workspace_revision':store.revision,'workspace':store.public}
                    if item!=expected: raise ValueError('backup source/workspace identity mismatch')
                elif kind=='trailer':
                    if item!={'type':'trailer','counts':counts,'sha256':digest.hexdigest()}: raise ValueError('backup trailer digest/count mismatch')
                    trailer=True;continue
                elif kind=='record':
                    if set(item)!={'type','record'}:raise ValueError('invalid record envelope')
                    record=item['record'];_validate_record(store,record)
                    prior=stage.execute('SELECT MAX(revision) FROM records WHERE record_id=?',(record['record_id'],)).fetchone()[0] or 0
                    if record['revision']!=prior+1:raise ValueError('backup revision gap or unordered history')
                    stage.execute('INSERT INTO records(record_id,revision,body) VALUES(?,?,?)',(record['record_id'],record['revision'],canonical(record)))
                    counts['records']+=1
                elif kind=='legacy':
                    if set(item)!={'type','legacy_id','context','sha256','raw_base64'}: raise ValueError('invalid legacy envelope')
                    raw_legacy=base64.b64decode(item['raw_base64'],validate=True)
                    if len(raw_legacy)>16384 or hashlib.sha256(raw_legacy).hexdigest()!=item['sha256']: raise ValueError('legacy raw digest mismatch')
                    converted=legacy_record(raw_legacy,item['context'])
                    if item['legacy_id']!=converted['record_id']+':'+str(converted['revision']): raise ValueError('legacy identity mismatch')
                    stage.execute('INSERT INTO legacy VALUES(?,?)',(item['legacy_id'],canonical(item)));counts['legacy']+=1
                elif kind=='progress':
                    if set(item)!={'type','progress'}:raise ValueError('invalid progress envelope')
                    progress=item['progress'];request=ProgressRequest.model_validate({k:v for k,v in progress.items() if k!='updated_at'})
                    if set(progress)!=set(request.model_dump())|{'updated_at'} or datetime.fromisoformat(progress['updated_at']).tzinfo is None:raise ValueError('invalid progress timestamp')
                    store._check_revision(request.workspace_revision)
                    if request.position_seconds>store._source(request.source_id)['duration_seconds']:raise ValueError('progress exceeds source')
                    stage.execute('INSERT INTO progress VALUES(?,?)',(request.source_id,canonical(progress)));counts['progress']+=1
                else:raise ValueError('unknown backup entry')
                digest.update(raw)
            if not trailer:raise ValueError('incomplete backup; trailer required')
            for lid, in stage.execute('SELECT legacy_id FROM legacy'):
                rid,revision=lid.rsplit(':',1)
                row=stage.execute('SELECT body FROM records WHERE record_id=? AND revision=?',(rid,int(revision))).fetchone()
                if not row or parse_json(row[0])['origin']!='legacy': raise ValueError('orphan legacy raw history')
            added=unchanged=0
            with store._db(write=True) as db:
                for lid,body in stage.execute('SELECT legacy_id,body FROM legacy'):
                    old=db.execute('SELECT body FROM legacy WHERE legacy_id=?',(lid,)).fetchone()
                    if old and old[0]!=body:raise WorkbenchConflict('divergent legacy history')
                    if not old: db.execute('INSERT INTO legacy VALUES(?,?)',(lid,body))
                for rid,revision,body in stage.execute('SELECT record_id,revision,body FROM records ORDER BY seq'):
                    record=parse_json(body)
                    if record['origin']=='legacy':
                        raw=db.execute('SELECT body FROM legacy WHERE legacy_id=?',(rid+':'+str(revision),)).fetchone()
                        if not raw: raise ValueError('legacy record has no exact raw history')
                        entry=parse_json(raw[0]);converted=legacy_record(base64.b64decode(entry['raw_base64']),entry['context'])
                        converted['created_at']=record['created_at']
                        if canonical(converted)!=body: raise ValueError('legacy transformation mismatch')
                    old=db.execute('SELECT body FROM revisions WHERE record_id=? AND revision=?',(rid,revision)).fetchone()
                    if old:
                        if old[0]!=body:raise WorkbenchConflict('divergent annotation revision')
                        unchanged+=1;continue
                    prior=db.execute('SELECT body FROM revisions WHERE record_id=? ORDER BY revision DESC LIMIT 1',(rid,)).fetchone()
                    previous=parse_json(prior[0]) if prior else None
                    if revision!=(previous['revision']+1 if previous else 1):raise ValueError('restore history gap')
                    if previous and any(record[k]!=previous[k] for k in ('source_id','source_sha256','kind','created_at')):raise ValueError('immutable entity identity changed')
                    db.execute('INSERT INTO revisions(record_id,revision,source_id,body) VALUES(?,?,?,?)',(rid,revision,record['source_id'],body));added+=1
                for sid,body in stage.execute('SELECT source_id,body FROM progress'):
                    old=db.execute('SELECT body FROM progress WHERE source_id=?',(sid,)).fetchone()
                    if not old or parse_json(body)['updated_at']>parse_json(old[0])['updated_at']:
                        db.execute('INSERT INTO progress VALUES(?,?) ON CONFLICT(source_id) DO UPDATE SET body=excluded.body',(sid,body))
                store._validate_references(db)
            return {'added':added,'unchanged':unchanged,'legacy':counts['legacy'],'progress':counts['progress']}
        except (sqlite3.Error,KeyError,TypeError) as exc: raise ValueError('invalid backup structure or duplicate entries') from exc
        finally:stage.close()

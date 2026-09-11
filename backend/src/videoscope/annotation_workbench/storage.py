"""SQLite append-only revisions with optimistic concurrency and reference validation."""
import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter
from .schema import RecordRequest, ProgressRequest, DATA_MODELS, reviewed_complete, PublicWorkspace
from .source_library import (WorkbenchConflict, WorkbenchMissing, WorkbenchForbidden,
                             canonical, read_json, parse_json, attest, safe_path)

REQUEST = TypeAdapter(RecordRequest)

def now(): return datetime.now(timezone.utc).isoformat()

class WorkbenchStore:
    def __init__(self, root: Path):
        self.root=safe_path(root)
        self.manifest=read_json(safe_path(self.root/'workspace.json'))
        self.manifest_hash=hashlib.sha256(canonical(self.manifest)).hexdigest()
        self.public=PublicWorkspace.model_validate(self.manifest['public']).model_dump()
        self.revision=hashlib.sha256(canonical(self.public)).hexdigest()
        if self.manifest['workspace_revision']!=self.revision: raise WorkbenchConflict('workspace identity is invalid')
        self.sources={s['source_id']:s for s in self.public['sources']}
        if len(self.sources)!=len(self.public['sources']): raise WorkbenchConflict('duplicate source identity')
        self.db_path=self.root/'annotations.sqlite3'
        for name in ('annotations.sqlite3','annotations.sqlite3-wal','annotations.sqlite3-shm'):
            if (self.root/name).is_symlink(): raise WorkbenchConflict('database symlink forbidden')
        with self._db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS revisions (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL,
              revision INTEGER NOT NULL, source_id TEXT NOT NULL, body BLOB NOT NULL,
              UNIQUE(record_id,revision));
            CREATE INDEX IF NOT EXISTS source_revisions ON revisions(source_id,record_id,revision);
            CREATE TABLE IF NOT EXISTS progress (source_id TEXT PRIMARY KEY, body BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS legacy (legacy_id TEXT PRIMARY KEY, body BLOB NOT NULL);
            CREATE TRIGGER IF NOT EXISTS revisions_no_update BEFORE UPDATE ON revisions
              BEGIN SELECT RAISE(ABORT,'immutable revision'); END;
            CREATE TRIGGER IF NOT EXISTS revisions_no_delete BEFORE DELETE ON revisions
              BEGIN SELECT RAISE(ABORT,'immutable revision'); END;
            CREATE TRIGGER IF NOT EXISTS legacy_no_update BEFORE UPDATE ON legacy
              BEGIN SELECT RAISE(ABORT,'immutable legacy bytes'); END;
            CREATE TRIGGER IF NOT EXISTS legacy_no_delete BEFORE DELETE ON legacy
              BEGIN SELECT RAISE(ABORT,'immutable legacy bytes'); END;
            ''')
            if db.execute('PRAGMA quick_check').fetchone()[0]!='ok': raise WorkbenchConflict('annotation database failed integrity check')
        # Only authorized descriptors are inspected; no reserve paths are persisted.
        for sid,source in self.sources.items():
            if source['review_allowed']:
                fd=self.media(sid); os.close(fd)

    @contextmanager
    def _db(self,write=False):
        try:
            for name in ('annotations.sqlite3','annotations.sqlite3-wal','annotations.sqlite3-shm','annotations.sqlite3-journal'):
                safe_path(self.root/name)
            db=sqlite3.connect(self.db_path,timeout=10)
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('PRAGMA synchronous=FULL')
            if write: db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                if write: db.commit()
            except BaseException:
                db.rollback(); raise
            finally: db.close()
        except sqlite3.DatabaseError as exc:
            raise WorkbenchConflict('annotation database is unavailable or inconsistent') from exc

    def check_manifest(self):
        if hashlib.sha256(canonical(read_json(safe_path(self.root/'workspace.json')))).hexdigest()!=self.manifest_hash:
            raise WorkbenchConflict('workspace changed; reopen the original workspace')

    def _source(self,sid,allow_reserved=False):
        self.check_manifest()
        source=self.sources.get(sid)
        if source is None: raise WorkbenchMissing('unknown source')
        if not allow_reserved and source['review_allowed'] is not True: raise WorkbenchForbidden('source is not authorized for content review')
        return source

    def _check_revision(self,value):
        if value!=self.revision: raise WorkbenchConflict('workspace revision changed')

    def workspace(self):
        self.check_manifest()
        with self._db() as db:
            rows=[parse_json(r[0]) for r in db.execute('SELECT body FROM progress')]
        return {**self.public,'workspace_revision':self.revision,'progress':max(rows,key=lambda r:r['updated_at']) if rows else None}

    @staticmethod
    def _latest(db,source_id=None):
        query='SELECT body FROM revisions r WHERE revision=(SELECT MAX(revision) FROM revisions WHERE record_id=r.record_id)'
        if source_id: query+=' AND source_id=?'
        return [parse_json(row[0]) for row in db.execute(query,(source_id,) if source_id else ())]

    def source(self,sid):
        source=self._source(sid)
        with self._db() as db:
            records=[r for r in self._latest(db,sid) if not r['archived']]
            row=db.execute('SELECT body FROM progress WHERE source_id=?',(sid,)).fetchone()
        return {'source':source,'records':records,'progress':parse_json(row[0]) if row else None}

    def history(self,rid):
        with self._db() as db:
            rows=[parse_json(r[0]) for r in db.execute('SELECT body FROM revisions WHERE record_id=? ORDER BY revision',(rid,))]
        if not rows: raise WorkbenchMissing('unknown record')
        self._source(rows[0]['source_id'])
        return rows

    def _validate_record(self,record,*,legacy=False):
        source=self._source(record['source_id'])
        if record['source_sha256']!=source['sha256']: raise ValueError('record source hash mismatch')
        kind=record['kind']
        if not record['record_id'].startswith(kind+'-'): raise ValueError('record identifier/kind mismatch')
        data=DATA_MODELS[kind].model_validate(record['data']).model_dump()
        duration=source['duration_seconds']
        for key,val in data.items():
            if key.endswith('_seconds') and val is not None and val>duration: raise ValueError('time exceeds source duration')
        lv=record.get('legacy',{}).get('schema_version') if legacy else None
        if record['status']=='human_reviewed' and not reviewed_complete(kind,data,legacy_version=lv): raise ValueError('reviewed record requires complete explicit answers')
        return data

    def _validate_references(self,db):
        rows={r['record_id']:r for r in self._latest(db)}
        for record in rows.values():
            if record['archived']: continue
            data=record['data']; sid=record['source_id']
            def ref(value,kind):
                if value is None: return None
                target=rows.get(value)
                if not target or target['archived'] or target['kind']!=kind or target['source_id']!=sid:
                    raise ValueError('reference must be a live same-source '+kind)
                return target['data']
            for key in ('team_id',): ref(data.get(key),'team')
            for key in ('actor_id','receiver_id','passer_id','incoming_player_id','outgoing_player_id'): ref(data.get(key),'player')
            ref(data.get('event_id'),'event')
            poss=ref(data.get('possession_id'),'possession')
            for p in data.get('points',[]): ref(p.get('player_id'),'player')
            if poss and record['kind']=='event':
                for key in ('start_seconds','end_seconds'):
                    value=data.get(key)
                    if value is None: continue
                    if poss.get('start_seconds') is not None and value<poss['start_seconds']: raise ValueError('event precedes possession')
                    if poss.get('end_seconds') is not None and value>poss['end_seconds']: raise ValueError('event exceeds possession')
        for row in db.execute('SELECT body FROM progress'):
            progress=parse_json(row[0]); selected=progress.get('selected_record_id')
            if selected is not None:
                target=rows.get(selected)
                # Archiving does not delete evidence or invalidate historical selection.
                if not target or target['source_id']!=progress['source_id']: raise ValueError('invalid progress selection')

    def save(self,payload):
        request=REQUEST.validate_python(payload)
        self._check_revision(request.workspace_revision)
        source=self._source(request.source_id)
        with self._db(write=True) as db:
            row=db.execute('SELECT body FROM revisions WHERE record_id=? ORDER BY revision DESC LIMIT 1',(request.record_id,)).fetchone()
            previous=parse_json(row[0]) if row else None
            if request.expected_revision!=(previous['revision'] if previous else 0): raise WorkbenchConflict('stale revision; preserve changes and reload history')
            if previous and (previous['source_id']!=request.source_id or previous['kind']!=request.kind): raise WorkbenchConflict('record source/kind cannot change')
            stamp=now()
            record=dict(record_id=request.record_id,revision=request.expected_revision+1,source_id=request.source_id,source_sha256=source['sha256'],kind=request.kind,status=request.status,archived=request.archived,data=request.data.model_dump(),created_at=previous['created_at'] if previous else stamp,updated_at=stamp,origin='human')
            if previous and 'legacy' in previous: record['legacy']=previous['legacy']
            record['data']=self._validate_record(record)
            record['needs_review']=record['status']!='human_reviewed'
            db.execute('INSERT INTO revisions(record_id,revision,source_id,body) VALUES(?,?,?,?)',(record['record_id'],record['revision'],record['source_id'],canonical(record)))
            self._validate_references(db)
        return record

    def progress(self,payload):
        request=ProgressRequest.model_validate(payload)
        self._check_revision(request.workspace_revision)
        source=self._source(request.source_id)
        if request.position_seconds>source['duration_seconds']: raise ValueError('position exceeds source duration')
        result={**request.model_dump(),'updated_at':now()}
        with self._db(write=True) as db:
            db.execute('INSERT INTO progress VALUES(?,?) ON CONFLICT(source_id) DO UPDATE SET body=excluded.body',(request.source_id,canonical(result)))
            self._validate_references(db)
        return result

    def media(self,sid):
        source=self._source(sid)
        binding=self.manifest['bindings'][sid]
        fd,_=attest(Path(binding['path']),source['sha256'],source['byte_size'],binding['identity'])
        return fd

    def export_lines(self):
        from .transfer import export_lines
        return export_lines(self)

    def restore_lines(self,lines):
        from .transfer import restore_lines
        return restore_lines(self,lines)

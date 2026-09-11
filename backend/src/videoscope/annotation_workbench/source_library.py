"""Private source bindings and content-attested descriptors."""
import hashlib
import json
import math
from fractions import Fraction
import os
import stat
from pathlib import Path

class WorkbenchConflict(ValueError):
    pass
class WorkbenchMissing(ValueError):
    pass
class WorkbenchForbidden(ValueError):
    pass

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def parse_json(raw):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def read_json(path,limit=4*1024*1024):
    with path.open('rb') as f: raw=f.read(limit+1)
    if len(raw)>limit: raise ValueError('JSON exceeds bound')
    return parse_json(raw)

def identity(s):
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def safe_path(path):
    # absolute() retains symlinks for explicit rejection; /tmp itself may be a platform alias.
    path=Path(path).absolute()
    for p in (path,*path.parents):
        if p.is_symlink(): raise WorkbenchConflict('source binding contains a symlink')
    return path

def attest(path,digest,size,expected=None):
    """Return the same open descriptor that was verified, never reopen for serving."""
    fd=None
    try:
        path=safe_path(path)
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size!=size: raise WorkbenchConflict('source size/type changed')
        if expected is not None and identity(st)!=expected: raise WorkbenchConflict('source identity changed')
        if expected is None:
            h=hashlib.sha256()
            while b:=os.read(fd,8*1024*1024): h.update(b)
            if h.hexdigest()!=digest: raise WorkbenchConflict('source content does not match audit')
            os.lseek(fd,0,os.SEEK_SET)
        if identity(os.fstat(fd))!=identity(st) or identity(path.stat())!=identity(st): raise WorkbenchConflict('source changed during attestation')
        return fd,identity(st)
    except (OSError,ValueError) as exc:
        if fd is not None: os.close(fd)
        if isinstance(exc,WorkbenchConflict): raise
        raise WorkbenchConflict('source unavailable or changed') from exc


def catalog_from_audit(audit_dir,project_root):
    inventory=read_json(audit_dir/'inventory.json')['sources']
    roles=read_json(audit_dir/'source-roles.json')['sources']
    rights=read_json(audit_dir/'source-rights-ledger.json')['sources']
    ids=[s['source_id'] for s in inventory]
    if len(ids)!=len(set(ids)) or len(ids)>100 or not ids: raise ValueError('invalid source catalog')
    if {s['source_id'] for s in roles}!=set(ids) or len(roles)!=len(ids): raise ValueError('roles membership mismatch')
    if {s['source_id'] for s in rights}!=set(ids) or len(rights)!=len(ids): raise ValueError('rights membership mismatch')
    roles={s['source_id']:s for s in roles}; rights={s['source_id']:s for s in rights}
    public=[]; bindings={}
    for i,s in enumerate(inventory,1):
        role=roles[s['source_id']]; right=rights[s['source_id']]
        if right.get('source_sha256')!=s['source_sha256']: raise ValueError('rights/source hash mismatch')
        allowed=role.get('content_decode_allowed_current_slice') is True
        if allowed and role['role']!='development_review': raise ValueError('unexpected source authorization role')
        sid=f'uba-{i:02d}'
        public.append(dict(source_id=sid,title=f'Матч {i:02d}',sha256=s['source_sha256'],duration_seconds=s['duration_seconds'],byte_size=s['bytes'],review_allowed=allowed,role=role['role'],source_group=s.get('grouping',{}).get('source_group_id',s['source_id']),training_rights='unknown',media_url=f'/media/{sid}' if allowed else None))
        streams=s.get('video_streams',[])
        fps=None
        if streams:
            for value in (streams[0].get('avg_frame_rate'),streams[0].get('r_frame_rate')):
                try:
                    number=float(Fraction(str(value)))
                    if math.isfinite(number) and 0<number<=1000: fps=number;break
                except (ValueError,ZeroDivisionError): pass
        public[-1]['fps']=fps
        from .schema import PublicSource
        public[-1]=PublicSource.model_validate(public[-1]).model_dump()
        if not allowed: continue  # Do not stat, resolve, hash or open reserved paths.
        rel=Path(s['source_relpath'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('source path must be project-relative')
        path=safe_path(project_root/rel)
        fd,attestation=attest(path,s['source_sha256'],s['bytes'])
        os.close(fd)
        bindings[sid]={'path':str(path),'identity':attestation}
    return public,bindings

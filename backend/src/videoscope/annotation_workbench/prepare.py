"""Initialize a new private annotation workspace without production import."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from .source_library import canonical, catalog_from_audit, safe_path


def prepare_workspace(*,audit_dir:Path,project_root:Path,legacy_batch:Path,output:Path,code_sha:str)->dict:
    output=Path(output).absolute()
    if output.exists() or output.is_symlink(): raise ValueError('workspace output already exists')
    project_root=safe_path(project_root)
    public,bindings=catalog_from_audit(Path(audit_dir),project_root)
    identity={'schema_version':1,'workspace_id':'uba-workbench-v1','title':'UBA · Разметка матчей','sources':public}
    revision=hashlib.sha256(canonical(identity)).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.workbench-',dir=output.parent))
    try:
        manifest={'public':identity,'workspace_revision':revision,'bindings':bindings,'code_sha':code_sha}
        (staging/'workspace.json').write_bytes(canonical(manifest)+b'\n')
        os.chmod(staging/'workspace.json',0o600)
        from .storage import WorkbenchStore
        from .legacy import import_legacy
        store=WorkbenchStore(staging)
        counts=import_legacy(store,Path(legacy_batch))
        os.rename(staging,output)
        return {'workspace_id':identity['workspace_id'],'workspace_revision':revision,'source_count':len(public),'review_allowed_count':len(bindings),**counts}
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('audit','project-root','legacy-batch','output'): parser.add_argument('--'+option,type=Path,required=True)
    parser.add_argument('--code-sha',default='unknown')
    args=parser.parse_args()
    try:
        result=prepare_workspace(audit_dir=args.audit,project_root=args.project_root,legacy_batch=args.legacy_batch,output=args.output,code_sha=args.code_sha)
    except (ValueError,OSError): parser.exit(1,'Workspace preparation failed; verify audit, source identity and legacy history.\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__': main()

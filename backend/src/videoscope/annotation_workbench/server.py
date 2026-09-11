"""Loopback-only full-match workbench, with no production Runtime or providers."""
import argparse
import asyncio
import tempfile
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from .storage import WorkbenchStore
from .source_library import WorkbenchConflict, WorkbenchForbidden, WorkbenchMissing, parse_json
from .streaming import media_response
from .transfer import MAX_BACKUP_BYTES

MAX_JSON_BYTES=64*1024
WEB_ROOT=Path(__file__).parent/'web'
SECURITY_HEADERS={
 'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
 'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer','Cache-Control':'no-store','Cross-Origin-Resource-Policy':'same-origin',
}


def create_app(workspace_dir:Path,*,port:int=8767)->FastAPI:
    if type(port) is not int or not 1<=port<=65535:raise ValueError('invalid loopback port')
    store=WorkbenchStore(workspace_dir)
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    app.state.store=store
    hosts={f'127.0.0.1:{port}',f'localhost:{port}',f'[::1]:{port}'}
    origins={'http://'+h for h in hosts}

    @app.middleware('http')
    async def boundary(request:Request,call_next):
        host=request.headers.get('host','');origin=request.headers.get('origin')
        if host not in hosts or (origin is not None and origin not in origins) or (request.method not in ('GET','HEAD','OPTIONS') and origin not in origins):
            response=JSONResponse({'detail':'untrusted loopback Host or Origin'},status_code=403)
        else:
            try:response=await call_next(request)
            except (ValueError,OSError):response=JSONResponse({'detail':'workspace data is invalid or unavailable'},status_code=409)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.exception_handler(WorkbenchConflict)
    async def conflict(request,exc):return JSONResponse({'detail':str(exc)},status_code=409)
    @app.exception_handler(WorkbenchForbidden)
    async def forbidden(request,exc):return JSONResponse({'detail':str(exc)},status_code=403)
    @app.exception_handler(WorkbenchMissing)
    async def missing(request,exc):return JSONResponse({'detail':str(exc)},status_code=404)
    @app.exception_handler(ValueError)
    async def invalid(request,exc):return JSONResponse({'detail':'Проверьте обязательные ответы, границы эпизода и ссылки на участников. Для восстановления нужна полная копия разметки тех же матчей.'},status_code=422)

    async def json_body(request):
        from fastapi import HTTPException
        if request.headers.get('content-type','').split(';')[0].strip()!='application/json':raise HTTPException(415,'application/json required')
        body=bytearray()
        async for chunk in request.stream():
            if len(body)+len(chunk)>MAX_JSON_BYTES:raise HTTPException(413,'JSON request exceeds bound')
            body.extend(chunk)
        value=parse_json(bytes(body))
        if not isinstance(value,dict):raise ValueError('JSON object required')
        return value

    @app.get('/api/health')
    def health():
        store.check_manifest()
        return {'status':'ready','workspace_id':store.public['workspace_id'],'schema_version':1}
    @app.get('/api/workspace')
    def workspace():return store.workspace()
    @app.get('/api/sources/{source_id}')
    def source(source_id:str):return store.source(source_id)
    @app.get('/api/records/{record_id}/history')
    def history(record_id:str):return store.history(record_id)
    @app.post('/api/records')
    async def save(request:Request):return await asyncio.to_thread(store.save,await json_body(request))
    @app.post('/api/progress')
    async def progress(request:Request):return await asyncio.to_thread(store.progress,await json_body(request))
    @app.api_route('/media/{source_id}',methods=['GET','HEAD'])
    def media(source_id:str,request:Request):
        return media_response(store.media(source_id),request.headers.get('range'),head=request.method=='HEAD')
    @app.get('/api/export')
    def export():return StreamingResponse(store.export_lines(),media_type='application/x-ndjson',headers={'Content-Disposition':'attachment; filename="videoscope-annotations.ndjson"'})
    @app.post('/api/restore')
    async def restore(request:Request):
        from fastapi import HTTPException
        if request.headers.get('content-type','').split(';')[0].strip()!='application/x-ndjson':raise HTTPException(415,'application/x-ndjson required')
        # Receive into a contained spool, with limits applied before parsing every line.
        with tempfile.TemporaryFile(dir=store.root) as backup:
            total=0;line_length=0
            async for chunk in request.stream():
                total+=len(chunk)
                if total>MAX_BACKUP_BYTES:raise HTTPException(413,'backup exceeds bound')
                for part in chunk.splitlines(keepends=True):
                    line_length+=len(part)
                    if line_length>128*1024:raise HTTPException(413,'backup line exceeds bound')
                    if part.endswith(b'\n'):line_length=0
                backup.write(chunk)
            backup.seek(0)
            return await asyncio.to_thread(store.restore_lines,backup)
    @app.get('/')
    def index():return FileResponse(WEB_ROOT/'index.html',media_type='text/html')
    app.mount('/',StaticFiles(directory=WEB_ROOT,check_dir=False),name='workbench-web')
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True);parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.workspace,port=args.port),host='127.0.0.1',port=args.port)

if __name__=='__main__':main()

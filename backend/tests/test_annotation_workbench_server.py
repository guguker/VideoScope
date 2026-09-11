import json
import os

import pytest
from fastapi.testclient import TestClient
from test_annotation_workbench_storage import workspace_fixture, shot


@pytest.fixture
def client(tmp_path):
    store,source,_=workspace_fixture(tmp_path)
    from videoscope.annotation_workbench.server import create_app
    with TestClient(create_app(store.root,port=8767),base_url='http://127.0.0.1:8767') as client:
        yield client,store,source


def test_health_and_attested_range_head_media(client):
    c,s,p=client
    assert c.get('/api/health').json()['status']=='ready'
    response=c.get('/media/uba-01',headers={'Range':'bytes=0-15'})
    assert response.status_code==206
    assert response.content==bytes(range(16))
    assert response.headers['content-range']=='bytes 0-15/256'
    assert c.head('/media/uba-01',headers={'Range':'bytes=-4'}).headers['content-length']=='4'
    assert c.get('/media/uba-02').status_code==403
    assert c.get('/api/sources/uba-02').status_code==403
    p.write_bytes(b'changed')
    assert c.get('/media/uba-01').status_code==409


def test_unknown_host_origin_json_bounds_and_no_paths(client):
    c,s,_=client
    assert c.get('/api/workspace',headers={'Host':'evil.example'}).status_code==403
    payload=dict(schema_version=1,workspace_revision=s.revision,source_id='uba-01',record_id='event-'+'a'*32,expected_revision=0,kind='event',status='human_reviewed',archived=False,data=shot())
    assert c.post('/api/records',json=payload).status_code==403
    assert c.post('/api/records',json=payload,headers={'Origin':'https://evil.example'}).status_code==403
    headers={'Origin':'http://127.0.0.1:8767'}
    assert c.post('/api/records',json=payload,headers=headers).status_code==200
    assert c.post('/api/records',json=payload,headers=headers).status_code==409
    assert c.post('/api/records',content='{',headers={**headers,'Content-Type':'application/json'}).status_code==422
    assert c.post('/api/records',content='x'*70000,headers={**headers,'Content-Type':'application/json'}).status_code==413
    assert c.post('/api/records',content='{}',headers=headers).status_code==415
    export=c.get('/api/export')
    assert str(s.root.parent).encode() not in export.content
    assert c.post('/api/restore',content=export.content,headers={**headers,'Content-Type':'application/x-ndjson'}).json()['added']==0
    assert c.post('/api/restore',content=export.content.splitlines()[0]+b'\n',headers={**headers,'Content-Type':'application/x-ndjson'}).status_code==422


def test_stream_never_reopens_swapped_path(client):
    c,s,p=client
    fd=s.media('uba-01')
    p.rename(p.with_suffix('.old'))
    p.write_bytes(b'malicious replacement')
    try: assert os.read(fd,16)==bytes(range(16))
    finally: os.close(fd)
    assert c.head('/media/uba-01').status_code==409

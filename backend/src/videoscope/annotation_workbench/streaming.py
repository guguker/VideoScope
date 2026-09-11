"""HTTP byte ranges read from an already-attested descriptor."""
import os
import re
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse
from .source_library import identity


def media_response(fd,range_header=None,*,head=False):
    info=os.fstat(fd);size=info.st_size;expected=identity(info)
    start,end=0,size-1;status=200
    if range_header is not None:
        match=re.fullmatch(r'bytes=(\d*)-(\d*)',range_header)
        if not match or not any(match.groups()):
            os.close(fd);raise HTTPException(416,'invalid byte range',headers={'Content-Range':f'bytes */{size}'})
        a,b=match.groups()
        if a: start=int(a);end=min(int(b),size-1) if b else size-1
        else: start=max(0,size-int(b));end=size-1
        if start>=size or end<start or (not a and int(b)==0):
            os.close(fd);raise HTTPException(416,'range outside source',headers={'Content-Range':f'bytes */{size}'})
        status=206
    headers={'Accept-Ranges':'bytes','Content-Length':str(end-start+1),'Content-Type':'video/mp4'}
    if status==206:headers['Content-Range']=f'bytes {start}-{end}/{size}'
    if head:
        os.close(fd);return Response(status_code=status,headers=headers)
    def chunks():
        try:
            offset=start
            while offset<=end:
                if identity(os.fstat(fd))!=expected: raise OSError('attested source changed during playback')
                data=os.pread(fd,min(256*1024,end-offset+1),offset)
                if not data:raise OSError('attested source truncated')
                offset+=len(data);yield data
        finally:os.close(fd)
    return StreamingResponse(chunks(),status_code=status,headers=headers)

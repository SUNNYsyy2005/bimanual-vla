"""Compress large monitoring JSON without changing its data or cache semantics."""
import gzip


def compress_json_response(response, request):
    if response.status_code!=200 or response.direct_passthrough or response.mimetype!='application/json' or response.headers.get('Content-Encoding'):
        return response
    response.vary.add('Accept-Encoding')
    if request.accept_encodings['gzip']<=0:return response
    body=response.get_data()
    if len(body)<2048:return response
    packed=gzip.compress(body,compresslevel=3,mtime=0)
    if len(packed)<len(body):
        response.set_data(packed)
        response.headers['Content-Encoding']='gzip'
    return response

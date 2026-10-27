import gzip,json,subprocess,sys
from flask import Flask,jsonify,request
from server_4090.response_compression import compress_json_response


def test_compression_lossless_and_negotiated():
    app=Flask(__name__)
    @app.get('/')
    def view():return compress_json_response(jsonify(points=[{'step':i,'loss':.123} for i in range(1000)]),request)
    client=app.test_client();plain=client.get('/',headers={'Accept-Encoding':'identity'});packed=client.get('/',headers={'Accept-Encoding':'gzip'})
    assert 'Content-Encoding' not in plain.headers
    assert packed.headers['Content-Encoding']=='gzip'
    assert gzip.decompress(packed.data)==plain.data
    assert len(packed.data)<len(plain.data)/3
    assert 'Accept-Encoding' in packed.headers['Vary']
    assert 'Content-Encoding' not in client.get('/',headers={'Accept-Encoding':'gzip;q=0'}).headers


def test_log_reader_skips_rollout_and_symlink_payloads(tmp_path):
    from server_4090.rlinf_dashboard import NATIVE_LOG_READ
    (tmp_path/'train.log').write_text('training loss=1')
    (tmp_path/'rollouts').mkdir();(tmp_path/'rollouts'/'large.log').write_text('not a training log')
    (tmp_path/'cycle').symlink_to(tmp_path,target_is_directory=True)
    output=subprocess.check_output([sys.executable,'-c',NATIVE_LOG_READ,str(tmp_path)],text=True,timeout=3)
    result=json.loads(output)
    assert result['files']==['train.log'] and 'loss=1' in result['log']

"""Pruebas HTTP del estudio local, con el pipeline real y archivos temporales."""
import base64
import io
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image
from closet import web

@pytest.fixture
def server(tmp_path, monkeypatch):
    photo = tmp_path / 'foto.png'
    pixels = np.full((140, 120, 3), 230, dtype=np.uint8)
    pixels[20:120, 35:85] = [30, 55, 95]
    Image.fromarray(pixels).save(photo)
    monkeypatch.setattr(web, 'STORE', tmp_path / 'results')
    monkeypatch.setattr(web, 'SAMPLES', [photo])
    monkeypatch.setattr(web, 'JOBS', {})
    monkeypatch.setattr(web.red_neuronal, 'status', lambda: {'ready':False})
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(web, 'POOL', pool)
    http = ThreadingHTTPServer(('127.0.0.1', 0), web.Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{http.server_port}', photo
    http.shutdown()
    http.server_close()
    pool.shutdown(wait=True)


def request(url, payload=None, origin=None):
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_home_and_samples(server):
    base, _ = server
    code, page = request(base + '/')
    assert code == 200
    assert 'Guardar prenda en tu clóset'.encode() in page
    assert 'Combinar con prenda de tu clóset'.encode() in page
    assert 'Ajustes del análisis'.encode() not in page
    assert b'Segmentar prenda' not in page
    code, data = request(base + '/api/samples')
    assert code == 200 and json.loads(data)[0]['name'] == 'foto.png'
    code, thumb = request(base + '/sample/0')
    assert code == 200 and Image.open(io.BytesIO(thumb)).format == 'JPEG'


def test_input_validation_and_origin(server):
    base, _ = server
    for payload in ({'sample': 99}, {'sample': -1}, {'sample': 0, 'k': 100}, {'image': 'invalid base64'}, {'sample':0, 'size':9999}):
        assert request(base + '/api/segment', payload)[0] == 400
    assert request(base + '/api/segment', {'sample':0}, 'https://other.example')[0] == 403


def test_missing_resources_and_path_traversal(server):
    base, _ = server
    for path in ('/missing', '/api/jobs/missing', '/media/../../README.md', '/sample/-1'):
        assert request(base + path)[0] == 404


def test_hybrid_and_learning_are_scoped_to_profile(server, monkeypatch):
    base, _ = server
    monkeypatch.setattr(web.red_neuronal, 'segmentation_ready', lambda:True)
    monkeypatch.setattr(web.red_neuronal, 'clothing_labels', lambda rgb:(rgb[..., 0] < 100).astype(np.uint8))

    def analyze(owner):
        code, body = request(base + '/api/segment', {'sample':0, 'engine':'auto', 'owner':owner})
        assert code == 202
        job_id = json.loads(body)['id']
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            data = json.loads(request(base + '/api/jobs/' + job_id)[1])
            if data['status'] in ('done', 'error'):
                assert data['status'] == 'done', data
                return data
            time.sleep(.1)
        pytest.fail('El análisis no terminó a tiempo')

    owner = 'a' * 32
    initial = analyze(owner)
    assert initial['engine'] == 'hibrido'
    assert {'mascara_kmeans', 'mascara_u2net', 'mascara'} <= initial['images'].keys()
    assert initial['found'] and not initial['alerts']
    category = 'camisas' if initial['classification']['category'] != 'camisas' else 'pantalones'
    subtype = web.SUBTYPES[category][0]
    code, body = request(base + '/api/wardrobe/save', {
        'job_id':initial['id'], 'owner':owner, 'category':category, 'subtype':subtype,
    })
    assert code == 201 and json.loads(body)['correction_learned']
    learned = analyze(owner)['classification']
    assert learned['personalized'] and (learned['category'], learned['subtype']) == (category, subtype)
    assert not analyze('b' * 32)['classification'].get('personalized')


@pytest.mark.parametrize('upload', [False, True])
def test_processing_downloads_and_history(server, upload):
    base, photo = server
    payload = {'k':4, 'size':400, 'rotation':90}
    if upload:
        payload.update(image=base64.b64encode(photo.read_bytes()).decode(), name='subida.png')
    else:
        payload['sample'] = 0
    code, body = request(base + '/api/segment', payload)
    assert code == 202
    job_id = json.loads(body)['id']
    deadline = time.monotonic() + 30
    while True:
        code, body = request(base + '/api/jobs/' + job_id)
        result = json.loads(body)
        if result['status'] in ('done', 'error') or time.monotonic() > deadline:
            break
        time.sleep(.1)
    assert result['status'] == 'done', result
    assert result['found'] and result['coverage'] > 0
    assert result['preprocessing']['steps']['rotacion_manual_grados'] == 90
    assert 'color_nombre' in result['features']
    for name, path in result['images'].items():
        code, data = request(base + path)
        assert code == 200
        im = Image.open(io.BytesIO(data))
        assert im.format == 'PNG'
        if name == 'sin_fondo':
            assert im.mode == 'RGBA'
    code, data = request(base + '/media/' + job_id + '/resultados.zip')
    assert code == 200
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert {'resultado.json', 'mascara.png', 'sin_fondo.png', 'normalizada.png', 'histograma.png'} <= set(z.namelist())
        assert z.testzip() is None
    assert json.loads(request(base + '/api/jobs')[1])[0]['id'] == job_id
    assert 'pattern' in result['attributes'] and 'classification' in result
    saved_payload={'owner':'a'*32,'job_id':job_id,'name':'Mi prenda','category':'camisas','subtype':'Camiseta','color':'azul'}
    code, body=request(base+'/api/wardrobe/save',saved_payload)
    assert code==201 and json.loads(body)['category']=='camisas'
    assert json.loads(body)['subtype']=='Camiseta'
    assert json.loads(body)['attributes']['color_family']=='azul'
    garment_id = json.loads(body)['id']
    edit = {'owner':'a'*32, 'id':garment_id, 'color':'rosa'}
    assert request(base+'/api/wardrobe/color',{**edit,'owner':'b'*32})[0]==404
    assert request(base+'/api/wardrobe/color',{**edit,'color':'inexistente'})[0]==400
    assert request(base+'/api/wardrobe/color',edit)[0]==200
    rows=json.loads(request(base+'/api/wardrobe?owner='+'a'*32)[1])
    assert rows[0]['attributes']['color_family']=='rosa'
    saved_payload['category']='chaquetas'
    assert request(base+'/api/wardrobe/save',saved_payload)[0]==400
    saved_payload['subtype']=''
    assert request(base+'/api/wardrobe/save',saved_payload)[0]==201
    rows=json.loads(request(base+'/api/wardrobe?owner='+'a'*32)[1])
    assert len(rows)==1 and rows[0]['category']=='chaquetas'
    assert json.loads(request(base+'/api/wardrobe?owner='+'b'*32)[1])==[]
    saved_payload['category']=''
    code, body=request(base+'/api/wardrobe/save',saved_payload)
    assert code==201 and json.loads(body)['category']==''
    assert json.loads(body)['category_confirmed'] is False


def test_existing_catalog_and_exclusions(server, monkeypatch):
    base, photo = server
    monkeypatch.setattr(web, 'PHOTO_ROOT', photo.parent)
    monkeypatch.setattr(web, 'CATALOG', {'foto.png': {'id':'cam_001', 'categoria':'camisas', 'incluir':'no', 'motivo':'exclusion_manual', 'split':'prueba'}})
    row = json.loads(request(base + '/api/samples')[1])[0]
    assert row['record_id'] == 'cam_001' and row['category'] == 'camisas'
    assert row['included'] is False and row['split'] == 'prueba'
    code, error = request(base + '/api/segment', {'sample':0})
    assert code == 400 and 'excluida' in json.loads(error)['error']


def test_neural_mode_is_used_in_web(server,monkeypatch):
    from closet import red_neuronal as cnn
    from closet.segmentacion import ResultadoSegmentacion
    base,_=server
    monkeypatch.setattr(cnn,'status',lambda:{'ready':True,'training_ids':[]})
    def neural(rgb,region):
        mask=np.zeros(rgb.shape[:2],bool);mask[20:120,35:85]=True
        return ResultadoSegmentacion(mask,mask.astype('uint8'),2,1,{
            'engine':'cnn','classification':{'category':'camisas','review':False,'support':.9,'method':'CNN prueba'}})
    monkeypatch.setattr(cnn,'analyze',neural)
    monkeypatch.setattr(web,'classify',lambda *_:pytest.fail('El modo CNN no debe reclasificar con KNN'))
    code,body=request(base+'/api/segment',{'sample':0,'size':400,'engine':'cnn'})
    assert code==202
    job=json.loads(body);deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        result=json.loads(request(base+'/api/jobs/'+job['id'])[1])
        if result['status'] in ('done','error'):break
        time.sleep(.05)
    assert result['status']=='done',result
    assert result['classification']['category']=='camisas'
    assert result['engine']=='cnn' and result['segmentation']['engine']=='cnn'
    assert 'color_nombre' in result['features'] and result['images']['mascara']
    code,saved=request(base+'/api/wardrobe/save',{'owner':'c'*32,'job_id':job['id'],'category':'camisas'})
    assert code==201 and json.loads(saved)['classification']['method']=='CNN prueba'


def test_cnn_unavailable_and_invalid_modes_are_explicit(server):
    base,_=server
    for options in ({'engine':'invalid'},{'engine':'cnn'},{'engine':'clasico','region':'wrong'}):
        assert request(base+'/api/segment',{'sample':0,**options})[0]==400


def test_previous_all_option_now_analyzes_one_garment(server, monkeypatch):
    from closet import red_neuronal as cnn
    from closet.segmentacion import ResultadoSegmentacion

    base, _ = server
    monkeypatch.setattr(cnn, 'status', lambda: {'ready': True, 'training_ids': []})
    analyzed_regions = []

    def analyze(rgb, region):
        analyzed_regions.append(region)
        labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
        labels[20:65, 35:85] = 1
        labels[75:125, 40:80] = 2
        return ResultadoSegmentacion(labels == 1, labels, 3, 1, {
            'classification': {'category': 'camisas', 'review': False}
        })

    monkeypatch.setattr(cnn, 'analyze', analyze)
    monkeypatch.setattr(web, 'classify_print', lambda *_: {'label': 'Sin estampado'})
    code, body = request(base + '/api/segment', {'sample': 0, 'engine': 'cnn', 'region': 'all', 'size': 400})
    assert code == 202
    job_id = json.loads(body)['id']
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = json.loads(request(base + '/api/jobs/' + job_id)[1])
        if job['status'] in ('done', 'error'):
            break
        time.sleep(.05)
    assert job['status'] == 'done', job
    assert analyzed_regions == ['auto']
    assert len(job['garments']) == 1
    assert job['classification']['category'] == 'camisas'

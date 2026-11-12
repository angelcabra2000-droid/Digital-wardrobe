"""Interfaz local de Digital Wardrobe. Ejecutar: python -m closet.web."""
from __future__ import annotations
import argparse, base64, csv, io, json, mimetypes, sqlite3, threading, time, uuid, zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image, UnidentifiedImageError
from .config import PreprocesoConfig, SegmentacionConfig
from .imagen_io import guardar_imagen, listar_imagenes
from .pipeline import procesar_foto
from .segmentacion import imagen_clusters, recortar_fondo
from .caracteristicas import extraer_caracteristicas
from .clasificacion import classify, MODEL
from .atributos import characterize
from .estampados import classify_print
from . import red_neuronal

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / 'resultados' / 'estudio'
WEB = Path(__file__).resolve().parent / 'frontend' / 'gossip.html'
STUDIO = Path(__file__).resolve().parent / 'frontend' / 'index.html'
ASSETS = Path(__file__).resolve().parent / 'frontend' / 'closet_gossip' / 'imagenes'
JOBS: dict = {}
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1)
PHOTO_ROOT = ROOT / 'Aplicacion base de datos'
if not PHOTO_ROOT.is_dir():
    PHOTO_ROOT = ROOT / 'BasedeDatos'
SAMPLES = listar_imagenes(PHOTO_ROOT)
CATALOG = {}
CATALOG_PATH = ROOT / 'datos' / 'dataset.csv'
if CATALOG_PATH.is_file():
    with CATALOG_PATH.open(encoding='utf-8-sig', newline='') as stream:
        CATALOG = {r['ruta'].replace(chr(92), '/'): r for r in csv.DictReader(stream)}

def sample_info(index, path):
    try:
        relative = path.relative_to(PHOTO_ROOT).as_posix()
    except ValueError:
        relative = path.name
    record = CATALOG.get(relative, {})
    return {'id': str(index), 'name': path.name, 'category': record.get('categoria') or path.parent.name,
            'image': f'/sample/{index}', 'record_id': record.get('id', ''),
            'included': record.get('incluir', 'si').strip().lower() not in ('no','false','0'),
            'reason': record.get('motivo', ''), 'warnings': record.get('avisos', ''),
            'split': record.get('split', ''), 'group':record.get('grupo', ''), 'width': record.get('ancho', ''),
            'height': record.get('alto', ''), 'mask_sample': record.get('muestra_mascara') == 'si'}

def wardrobe_db():
    STORE.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STORE / 'closet.sqlite3', timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute('CREATE TABLE IF NOT EXISTS garments (id TEXT PRIMARY KEY, owner TEXT NOT NULL, job_id TEXT NOT NULL, name TEXT NOT NULL, category TEXT NOT NULL, created TEXT NOT NULL, data TEXT NOT NULL, UNIQUE(owner, job_id))')
    return connection

def validate_owner(owner):
    if not isinstance(owner, str) or len(owner) != 32 or any(c not in '0123456789abcdef' for c in owner):
        raise ValueError('Perfil local inválido')
    return owner

def update(job_id, **fields):
    with LOCK:
        JOBS[job_id].update(fields)

def save_images(job_id, images):
    relative = Path(job_id)
    paths = {}
    for name, array in images.items():
        path = relative / f'{name}.png'
        guardar_imagen(array, STORE / path)
        paths[name] = f'/media/{path.as_posix()}'
    return paths


def garment_result(job_id, result, common_images, record, engine):
    pre, seg, norm = result.preproceso, result.segmentacion, result.normalizada
    images = {'mascara': (seg.mascara * 255).astype(np.uint8),
              'sin_fondo': recortar_fondo(pre.original, seg.mascara)}
    features, attributes = {}, {}
    if norm is not None:
        images['normalizada'] = recortar_fondo(norm.rgb, norm.mascara)
        features = extraer_caracteristicas(norm).como_dict()
        attributes = characterize(norm, features, source_rgb=pre.original, source_mask=seg.mascara)
        attributes['print'] = classify_print(pre.original, seg.mascara)

    classification = seg.diagnostico.get('classification')
    if classification is None:
        classification = classify(features, record.get('record_id'), record.get('group')) if features else {
            'category': None, 'review': True, 'support': 0
        }
    elif record.get('record_id') in red_neuronal.status().get('training_ids', []):
        classification.update(in_training=True, review=True)
    if engine == 'cnn' and not seg.encontrada:
        classification.update(category=None, review=True, message='No se encontró ropa suficiente en la foto.')
    if record.get('category') and classification.get('category') and record['category'] != classification['category']:
        classification.update(review=True, catalog_category=record['category'], catalog_mismatch=True)

    return {
        'garment_id': '1',
        'found': bool(seg.encontrada),
        'coverage': round(float(seg.mascara.mean()) * 100, 1),
        'alerts': result.alertas,
        'features': features,
        'classification': classification,
        'attributes': attributes,
        'segmentation': seg.diagnostico,
        'images': {**common_images, **save_images(job_id, images)},
    }


def process(job_id, source, options):
    started = time.monotonic()
    folder = STORE / job_id
    try:
        update(job_id, status='processing', message='Preparando y segmentando la imagen…')
        engine = options.get('engine', 'clasico')
        region = options.get('region', 'auto')
        pre_config = PreprocesoConfig(lado_max=options['size'])
        result = procesar_foto(source, cfg_pre=pre_config,
                               cfg_seg=SegmentacionConfig(k=options['k'], peso_textura=options['texture']),
                               preprocesar=options['preprocess'], motor=engine, region=region)
        pre = result.preproceso
        common_images = save_images(job_id, {
            'original': pre.original,
            'preprocesada': pre.normalizada,
            'clusters': imagen_clusters(result.segmentacion, pre.original),
        })
        with LOCK:
            data = dict(JOBS[job_id])
        update(job_id, message='Midiendo las características de la prenda…')
        garment = garment_result(job_id, result, common_images, data.get('record') or {}, engine)
        data.update(garment, status='done', message='Análisis terminado',
                    garments=[garment] if garment['found'] else [],
                    seconds=round(time.monotonic() - started, 1),
                    width=int(pre.original.shape[1]), height=int(pre.original.shape[0]), engine=engine)
        (folder / 'resultado.json').write_text(json.dumps(data, ensure_ascii=False, default=float), encoding='utf-8')
        with zipfile.ZipFile(folder / 'resultados.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in folder.rglob('*'):
                if path.is_file() and path.name not in ('resultados.zip', 'entrada.png'):
                    archive.write(path, path.relative_to(folder).as_posix())
        with LOCK:
            JOBS[job_id] = data
    except Exception as exc:
        update(job_id, status='error', message=f'No se pudo procesar la imagen: {exc}')


class Handler(BaseHTTPRequestHandler):
    def send(self, data, content_type='application/json', status=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, default=float).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == '/':
                return self.send(WEB.read_bytes(), 'text/html; charset=utf-8')
            if path.startswith('/assets/'):
                target = (ASSETS / path.removeprefix('/assets/')).resolve()
                if not target.is_relative_to(ASSETS.resolve()) or target.suffix.lower() not in ('.jpg','.png') or not target.is_file():
                    return self.send({'error':'Imagen no encontrada'}, status=404)
                return self.send(target.read_bytes(), mimetypes.guess_type(target.name)[0])
            if path == '/estudio':
                return self.send(STUDIO.read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/model':
                neural = red_neuronal.status()
                if neural['ready']:
                    return self.send(neural)
                if not MODEL.exists():
                    return self.send({'ready':False})
                model = json.loads(MODEL.read_text())
                return self.send({'ready':True, 'engine':'clasico', 'method':'KNN · forma, color y textura', 'records':len(model['records']), 'validation':model['validation']})
            if path == '/api/wardrobe':
                owner = validate_owner(parse_qs(urlparse(self.path).query).get('owner',[''])[0])
                with wardrobe_db() as connection:
                    rows = connection.execute('SELECT data FROM garments WHERE owner=? ORDER BY created DESC', (owner,)).fetchall()
                return self.send([json.loads(r['data']) for r in rows])
            if path == '/api/samples':
                return self.send([sample_info(i, p) for i,p in enumerate(SAMPLES)])
            if path.startswith('/sample/'):
                index = int(path.split('/')[-1])
                if index < 0 or index >= len(SAMPLES):
                    raise ValueError('Imagen inexistente')
                with Image.open(SAMPLES[index]) as im:
                    from PIL import ImageOps
                    im = ImageOps.exif_transpose(im).convert('RGB')
                    im.thumbnail((420, 420))
                    buffer = io.BytesIO(); im.save(buffer, 'JPEG', quality=85)
                return self.send(buffer.getvalue(), 'image/jpeg')
            if path == '/api/jobs':
                with LOCK:
                    data = [dict(v) for v in reversed(list(JOBS.values()))]
                return self.send(data)
            if path.startswith('/api/jobs/'):
                with LOCK:
                    data = dict(JOBS[path.split('/')[-1]])
                return self.send(data)
            if path.startswith('/media/'):
                file = (STORE / path.removeprefix('/media/')).resolve()
                if not file.is_relative_to(STORE.resolve()) or not file.is_file() or file.suffix not in {'.png', '.json', '.zip'}:
                    return self.send({'error':'Archivo no encontrado'}, status=404)
                return self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
            return self.send({'error':'Ruta no encontrada'}, status=404)
        except (ValueError, KeyError, IndexError, FileNotFoundError):
            self.send({'error':'No se encontró el recurso'}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ('/api/segment','/api/wardrobe/save'):
            return self.send({'error':'Ruta no encontrada'}, status=404)
        # Only the local UI can start processing; no cross-origin browser requests.
        if self.headers.get('Origin') and self.headers['Origin'] != f'http://{self.headers.get("Host")}':
            return self.send({'error':'Origen no permitido'}, status=403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 28 * 1024 * 1024:
                raise ValueError('La imagen debe pesar menos de 20 MB')
            payload = json.loads(self.rfile.read(length))
            if path == '/api/wardrobe/save':
                owner = validate_owner(payload.get('owner'))
                with LOCK:
                    job = dict(JOBS[payload['job_id']])
                if job['status'] != 'done' or not job.get('found'):
                    raise ValueError('Primero analiza una prenda válida')
                garment_id = payload.get('garment_id', '1')
                if job.get('garments'):
                    garment = next((item for item in job['garments'] if item['garment_id'] == garment_id), None)
                    if garment is None:
                        raise ValueError('No se encontró esa prenda en el análisis')
                elif garment_id == '1':
                    garment = job
                else:
                    raise ValueError('No se encontró esa prenda en el análisis')
                category = payload.get('category') or garment.get('classification', {}).get('category')
                if category not in ('camisas','chaquetas','pantalones'):
                    raise ValueError('Elige una categoría para guardar la prenda')
                name = str(payload.get('name') or job['name']).strip()[:100]
                if not name:
                    raise ValueError('Escribe un nombre para la prenda')
                # Conserva la clave de las fichas antiguas y da una clave propia a cada recorte adicional.
                source_key = job['id'] if garment_id == '1' else f"{job['id']}:{garment_id}"
                with wardrobe_db() as connection:
                    connection.execute('BEGIN IMMEDIATE')
                    previous = connection.execute(
                        'SELECT id, created FROM garments WHERE owner=? AND job_id=?', (owner, source_key)
                    ).fetchone()
                    saved = {'id': previous['id'] if previous else uuid.uuid4().hex,
                             'job_id': job['id'], 'garment_id': garment_id,
                             'name': name, 'category': category,
                             'created': previous['created'] if previous else time.strftime('%Y-%m-%dT%H:%M:%S'),
                             'images': garment['images'], 'features': garment['features'],
                             'attributes': garment.get('attributes', {}),
                             'classification': garment.get('classification', {}), 'category_confirmed': True}
                    connection.execute(
                        'INSERT INTO garments (id,owner,job_id,name,category,created,data) VALUES (?,?,?,?,?,?,?) '
                        'ON CONFLICT(owner,job_id) DO UPDATE SET name=excluded.name, category=excluded.category, data=excluded.data',
                        (saved['id'], owner, source_key, name, category, saved['created'], json.dumps(saved, ensure_ascii=False)),
                    )
                return self.send(saved, status=201)
            options = {'k': int(payload.get('k', 4)), 'size': int(payload.get('size', 800)),
                       'texture': float(payload.get('texture', 3)), 'preprocess': bool(payload.get('preprocess', True)),
                       'engine': payload.get('engine', 'cnn' if red_neuronal.status()['ready'] else 'clasico'),
                       'region': payload.get('region', 'auto')}
            if options['region'] == 'all':
                options['region'] = 'auto'  # Compatibilidad con pestañas abiertas antes del cambio.
            if options['engine'] not in ('cnn', 'clasico') or options['region'] not in ('auto', 'upper', 'lower'):
                raise ValueError('Motor o región inválidos')
            if options['engine'] == 'cnn' and not red_neuronal.status()['ready']:
                raise ValueError('La red CNN no está disponible en este equipo. Instala requirements-cnn.txt y los pesos.')
            if options['k'] not in range(2,9) or options['size'] not in (400,600,800) or not 0 <= options['texture'] <= 6:
                raise ValueError('Parámetros fuera de rango')
            with LOCK:
                if sum(v['status'] in ('queued','processing') for v in JOBS.values()) >= 6:
                    return self.send({'error':'Hay varias imágenes en cola. Espera a que terminen.'}, status=429)
            job_id = uuid.uuid4().hex
            folder = STORE / job_id
            folder.mkdir(parents=True)
            if 'sample' in payload:
                index = int(payload['sample'])
                if index < 0 or index >= len(SAMPLES):
                    raise ValueError('Ejemplo inexistente')
                source = SAMPLES[index]
                info = sample_info(index, source)
                if not info['included']:
                    raise ValueError('Esta foto está excluida en el catálogo: ' + info['reason'])
                name = source.name
            else:
                raw = base64.b64decode(payload['image'], validate=True)
                if len(raw) > 20 * 1024 * 1024:
                    raise ValueError('La imagen debe pesar menos de 20 MB')
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width * im.height > 40_000_000:
                        raise ValueError('Usa una imagen de menos de 40 megapíxeles')
                    im.verify()
                source = folder / 'entrada.png'
                source.write_bytes(raw)
                name = Path(str(payload.get('name', 'Mi prenda'))).name[:150]
            data = {'id':job_id, 'name':name, 'status':'queued', 'message':'Esperando su turno…',
                    'created':time.strftime('%Y-%m-%dT%H:%M:%S'), 'options':options,
                    'record':sample_info(index, source) if 'sample' in payload else None}
            with LOCK:
                JOBS[job_id] = data
            POOL.submit(process, job_id, source, options)
            self.send(data, status=202)
        except (ValueError, KeyError, UnidentifiedImageError, OSError) as exc:
            self.send({'error':str(exc)}, status=400)

    def log_message(self, format, *args):
        print(format % args, flush=True)

def main():
    parser = argparse.ArgumentParser(description='Estudio visual de Digital Wardrobe')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1', choices=['127.0.0.1','0.0.0.0'])
    args = parser.parse_args()
    STORE.mkdir(parents=True, exist_ok=True)
    for p in sorted(STORE.glob('*/resultado.json'), key=lambda p:p.stat().st_mtime):
        try:
            data = json.loads(p.read_text()); JOBS[data['id']] = data
        except (ValueError, KeyError):
            pass
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Closet Gossip listo en http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

if __name__ == '__main__':
    main()

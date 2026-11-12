"""Clasificación KNN por características; sin redes neuronales.
Entrenar: python -m closet.clasificacion --entrenar
Solo usa filas incluidas del conjunto de entrenamiento del CSV.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from .caracteristicas import columnas_forma, extraer_caracteristicas
from .config import PreprocesoConfig
from .pipeline import procesar_foto

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / 'datos' / 'modelo_caracteristicas.json'
BLOCKS = [columnas_forma(), ['color_L', 'color_a', 'color_b_lab', 'n_colores'],
          ['har_contraste', 'har_homogeneidad', 'har_energia', 'har_entropia', 'detalle_frac']]
WEIGHTS = [.8, .08, .12]

def vector(features):
    return np.array([float(features.get(key, 0)) for block in BLOCKS for key in block], dtype=float)

def nearest(features, records, k=3):
    if not records:
        return {'category':None, 'support':0, 'review':True, 'neighbors':[]}
    x = np.stack([vector(r['features']) for r in records])
    q = vector(features)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1
    z = np.clip((x - mean) / std, -8, 8)
    query = np.clip((q - mean) / std, -8, 8)
    distances = np.zeros(len(records))
    start = 0
    for block, weight in zip(BLOCKS, WEIGHTS):
        end = start + len(block)
        distances += weight * ((z[:, start:end] - query[start:end]) ** 2).mean(axis=1)
        start = end
    order = np.argsort(distances)[:min(k, len(records))]
    votes = {}
    for i in order:
        label = records[i]['category']
        votes[label] = votes.get(label, 0) + 1 / max(float(distances[i]) ** .5, .05)
    label = max(votes, key=votes.get)
    support = votes[label] / sum(votes.values())
    return {'category':label, 'support':round(support, 3), 'review':support < .7,
            'method':'KNN · forma, color y textura',
            'neighbors':[{'id':records[i]['id'], 'category':records[i]['category']} for i in order]}

def classify(features, exclude_id=None, exclude_group=None):
    if not MODEL.exists():
        return {'category':None,'support':0,'review':True,'message':'Modelo de características no disponible'}
    model = json.loads(MODEL.read_text())
    records = [r for r in model['records'] if r['id'] != exclude_id and (not exclude_group or r['group'] != exclude_group)]
    result = nearest(features, records)
    result['validation'] = model.get('validation', {})
    return result

def describe_shape(features):
    ratio = float(features.get('ratio_aspecto', 1))
    if ratio >= 1.8:
        return 'Alargada'
    if ratio <= .9:
        return 'Ancha'
    return 'Intermedia'

def train(root=ROOT, output=MODEL):
    with (root/'datos/dataset.csv').open(encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if r['incluir'].lower() == 'si' and r['split'] == 'entrenamiento']
    records, skipped = [], []
    photos = root/'Aplicacion base de datos'
    if not photos.exists():
        photos = root/'BasedeDatos'
    cache = root/'resultados/entrenamiento_caracteristicas'
    cache.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        path = cache/(row['id']+'.json')
        try:
            if path.exists():
                features = json.loads(path.read_text())
            else:
                result = procesar_foto(photos/row['ruta'], cfg_pre=PreprocesoConfig(lado_max=600))
                if result.normalizada is None:
                    raise ValueError('sin prenda detectada')
                features = extraer_caracteristicas(result.normalizada).como_dict()
                path.write_text(json.dumps(features, default=float))
            records.append({'id':row['id'],'category':row['categoria'],'group':row['grupo'] or row['id'],'features':features})
        except Exception as exc:
            skipped.append({'id':row['id'],'reason':str(exc)})
        print(f'{index+1}/{len(rows)}: {row["id"]}', flush=True)
    if len({r['category'] for r in records}) != 3:
        raise ValueError('Faltan categorías para entrenar')
    correct, totals = {}, {}
    for r in records:
        others = [s for s in records if s['group'] != r['group']]
        pred = nearest(r['features'], others)
        label = r['category']
        totals[label] = totals.get(label, 0)+1
        correct[label] = correct.get(label, 0)+int(pred['category']==label)
    validation = {'method':'dejar fuera un grupo del entrenamiento', 'total':len(records),
                  'accuracy':round(sum(correct.values())/len(records), 4),
                  'by_category':{k:round(correct.get(k,0)/n, 4) for k,n in totals.items()},
                  'test_set_used':False}
    model = {'version':1,'training_size':600,'records':records,'skipped':skipped,'validation':validation}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model,ensure_ascii=False,default=float))
    print(json.dumps(validation,ensure_ascii=False), flush=True)
    return model

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--entrenar',action='store_true',required=True)
    parser.parse_args()
    train()

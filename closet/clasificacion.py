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
BLOCK_NAMES = ('forma', 'color', 'textura')
SUBTYPES = {
    'camisas': ('Camisa', 'Camiseta', 'Esqueleto', 'Polo'),
    'chaquetas': ('Chaqueta de jean', 'Buzo', 'Chaqueta de cuero', 'Blazer', 'Chaqueta acolchada'),
    'pantalones': ('Jean', 'Sudadera', 'Pantaloneta', 'Cargo', 'Leggings'),
}

def vector(features):
    return np.array([float(features.get(key, 0)) for block in BLOCKS for key in block], dtype=float)

def nearest(features, records, k=3, blocks=(0, 1, 2), max_distance=None, reference=None):
    if not records:
        return {'category':None, 'support':0, 'review':True, 'neighbors':[]}
    x = np.stack([vector(r['features']) for r in records])
    q = vector(features)
    scale = x if reference is None else np.stack([vector(r['features']) for r in reference])
    mean = scale.mean(axis=0)
    std = scale.std(axis=0)
    std[std < 1e-6] = 1
    z = np.clip((x - mean) / std, -8, 8)
    query = np.clip((q - mean) / std, -8, 8)
    distances = np.zeros(len(records))
    start = 0
    for index, (block, weight) in enumerate(zip(BLOCKS, WEIGHTS)):
        end = start + len(block)
        if index in blocks:
            distances += weight * ((z[:, start:end] - query[start:end]) ** 2).mean(axis=1)
        start = end
    distances = np.sqrt(distances / sum(WEIGHTS[i] for i in blocks))
    order = np.argsort(distances, kind='stable')[:min(k, len(records))]
    votes = {}
    for i in order:
        label = records[i]['category']
        votes[label] = votes.get(label, 0) + 1 / max(float(distances[i]), .05)
    label = max(votes, key=votes.get)
    support = votes[label] / sum(votes.values())
    distance = float(distances[order[0]])
    rejected = max_distance is not None and (support < .7 or distance > max_distance)
    return {'category':None if rejected else label, 'support':round(support, 3),
            'review':support < .7 or rejected, 'distance':distance,
            'message':'El KNN no encontró vecinos suficientemente cercanos o de acuerdo.' if rejected else None,
            'method':'KNN · ' + ', '.join(BLOCK_NAMES[i] for i in blocks),
            'neighbors':[{'id':records[i]['id'], 'category':records[i]['category']} for i in order]}

def classify(features, exclude_id=None, exclude_group=None, personal=()):
    if not MODEL.exists():
        return {'category':None,'support':0,'review':True,'message':'Modelo de características no disponible'}
    model = json.loads(MODEL.read_text(encoding='utf-8'))
    records = [r for r in model['records'] if r['id'] != exclude_id and (not exclude_group or r['group'] != exclude_group)]
    settings = model.get('settings', {})
    result = nearest(features, records, **settings)
    category = result['category']
    subtype_model = model.get('subtypes', {}).get(category, {})
    subtype_records = [{**row, 'category':row['subtype']} for row in records
                       if row['category'] == category and row.get('subtype') in subtype_model.get('labels', [])]
    result['subtype'] = None
    if len({row['category'] for row in subtype_records}) >= 2:
        prediction = nearest(features, subtype_records, **subtype_model['settings'])
        if len({row['group'] for row in subtype_records if row['category'] == prediction['category']}) >= 2:
            result['subtype'] = prediction['category']
        result['subtype_evidence'] = prediction
    result['validation'] = model.get('validation', {})
    if personal:
        close = nearest(features, personal, k=3, blocks=settings.get('blocks', (0, 1, 2)),
                        max_distance=.18, reference=model['records'])
        if close['category']:
            neighbors = {row['id']:row for row in personal}
            nearest_match = neighbors[close['neighbors'][0]['id']]
            subtype = nearest_match.get('subtype') if nearest_match['category'] == close['category'] else None
            return {**close, 'subtype':subtype, 'personalized':True, 'review':True,
                    'method':'KNN · correcciones de tu clóset',
                    'message':'Sugerencia basada en prendas similares que corregiste.'}
    return result


def refine_silhouette(prediction, mask):
    """Busca dos tirantes finos separados unidos a un torso continuo."""
    if prediction.get('personalized'):
        return prediction
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return prediction
    crop = mask[rows.min():rows.max()+1, cols.min():cols.max()+1]
    height, width = crop.shape
    if height < 30 or width < 25 or not .85 <= height / width <= 2.4:
        return prediction

    strap_rows = crop[int(.05*height):int(.22*height)]
    matches = 0
    for row in strap_rows:
        runs = np.flatnonzero(np.diff(np.r_[False, row, False])).reshape(-1, 2)
        runs = [(start, end) for start, end in runs if end-start >= .025*width]
        if len(runs) == 2:
            (a, b), (c, d) = runs
            if (b-a)+(d-c) <= .27*width and c-b >= .22*width:
                matches += 1
    torso = crop[int(.38*height):int(.88*height)]
    strap_fraction = matches / len(strap_rows)
    central_fill = float(torso[:, int(.35*width):int(.65*width)].mean())
    if strap_fraction < .75 or central_fill < .95 or float(torso.mean()) < .52:
        return prediction
    return {'category':'camisas', 'subtype':'Esqueleto', 'review':True,
            'method':'Geometría de tirantes sobre la máscara',
            'message':'Se detectaron dos tirantes finos sobre un torso continuo. Revisa la sugerencia.',
            'silhouette_evidence':{'strap_fraction':round(strap_fraction, 3), 'central_fill':round(central_fill, 3)},
            'knn_prediction':prediction}


def fit_settings(records, candidates=None):
    """Compara subconjuntos de bloques y valores de k dejando fuera grupos completos."""
    import itertools

    if candidates is None:
        candidates = [blocks for count in range(1, 4)
                      for blocks in itertools.combinations(range(3), count)]
    labels = sorted({row['category'] for row in records})
    folds = [(row, [other for other in records if other['group'] != row['group']])
             for row in records]
    scores = []
    for blocks in candidates:
        for k in (1, 3, 5, 7):
            if any(len(others) < k for _, others in folds):
                continue
            predictions = [nearest(row['features'], others, k, blocks) for row, others in folds]
            recalls = {label: np.mean([p['category'] == label for (row, _), p in zip(folds, predictions)
                                      if row['category'] == label]) for label in labels}
            scores.append({'blocks':blocks, 'k':k, 'features':sum(len(BLOCKS[i]) for i in blocks),
                           'balanced_accuracy':float(np.mean(list(recalls.values()))),
                           'accuracy':float(np.mean([p['category'] == row['category']
                                                      for (row, _), p in zip(folds, predictions)])),
                           'by_category':recalls})
    if not scores:
        raise ValueError('Se necesitan al menos dos grupos independientes')
    scores.sort(key=lambda row: (-row['balanced_accuracy'], row['features'], row['k']))
    best = scores[0]
    settings = {'blocks':best['blocks'], 'k':best['k']}
    distances = [nearest(row['features'], others, **settings)['distance'] for row, others in folds]
    settings['max_distance'] = float(np.quantile(distances, .95))
    return settings, scores

def train(root=ROOT, output=MODEL):
    with (root/'datos/dataset.csv').open(encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if r['incluir'].lower() == 'si' and r['split'] == 'entrenamiento']
    annotation_path = root/'datos/subcategorias.csv'
    annotations = {}
    if annotation_path.exists():
        with annotation_path.open(encoding='utf-8-sig', newline='') as stream:
            annotations = {r['id']:r['subcategoria'].strip() for r in csv.DictReader(stream)}
    records, skipped = [], []
    photos = root/'Aplicacion base de datos'
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
        else:
            subtype = annotations.get(row['id'])
            if subtype:
                if subtype not in SUBTYPES[row['categoria']]:
                    raise ValueError(f'Subcategoría inválida para {row["id"]}: {subtype}')
                records[-1]['subtype'] = subtype
        print(f'{index+1}/{len(rows)}: {row["id"]}', flush=True)
    if len({r['category'] for r in records}) != 3:
        raise ValueError('Faltan categorías para entrenar')
    settings, scores = fit_settings(records)
    best = scores[0]
    subtypes = {}
    for category, labels in SUBTYPES.items():
        category_records = [r for r in records if r['category'] == category and r.get('subtype')]
        eligible = {label for label in labels if len({r['group'] for r in category_records
                                                   if r['subtype'] == label}) >= 2}
        if len(eligible) >= 2:
            examples = [{**r, 'category':r['subtype']} for r in category_records if r['subtype'] in eligible]
            subtype_settings, subtype_scores = fit_settings(examples)
            subtypes[category] = {'settings':subtype_settings, 'labels':sorted(eligible),
                                  'validation':subtype_scores[0]}
    validation = {'method':'dejar fuera un grupo del entrenamiento', 'total':len(records),
                  'accuracy':round(best['accuracy'], 4), 'balanced_accuracy':best['balanced_accuracy'],
                  'by_category':best['by_category'], 'feature_count':best['features'],
                  'test_set_used':False}
    model = {'version':2,'training_size':600,'records':records,'skipped':skipped,'validation':validation,
             'settings':settings, 'experiments':scores, 'subtypes':subtypes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model,ensure_ascii=False,default=float), encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False), flush=True)
    return model

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--entrenar',action='store_true',required=True)
    parser.parse_args()
    train()

"""U²-Net de ropa + clasificador CNN ResNet-18. Inferencia local, sin descargas HTTP.

U²-Net es una variante de U-Net, no la arquitectura original de 2015.
Los pesos de segmentación son externos; la cabeza de clasificación se adapta al catálogo.
"""
from functools import lru_cache
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'datos' / 'red_neuronal'
CLASSES = ('camisas', 'chaquetas', 'pantalones')
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def status():
    metadata = MODELS / 'modelo.json'
    deps = all(importlib.util.find_spec(m) is not None for m in ('torch', 'torchvision', 'onnxruntime'))
    ready = deps and all((MODELS / p).is_file() for p in ('u2net_cloth_seg.onnx', 'clasificador.pt', 'modelo.json'))
    info = json.loads(metadata.read_text()) if metadata.is_file() else {}
    return {**info, 'ready': ready, 'engine': 'cnn', 'method': 'U²-Net + CNN ResNet-18'}


def tensor_image(image, size=224):
    """Misma geometría en entrenamiento e inferencia: conservar proporción y rellenar."""
    im = ImageOps.pad(image.convert('RGB'), (size, size), method=Image.Resampling.BILINEAR, color=(245, 245, 245))
    a = (np.asarray(im, np.float32) / 255 - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1))


@lru_cache(maxsize=1)
def classifier():
    import torch
    from torchvision.models import resnet18
    torch.set_num_threads(4)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    data = torch.load(MODELS / 'clasificador.pt', map_location='cpu', weights_only=True)
    if tuple(data['classes']) != CLASSES:
        raise ValueError('Las categorías de la CNN no coinciden con la aplicación')
    model.load_state_dict(data['state_dict'])
    return model.eval()


def classify_crop(rgb, mask):
    import torch
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {'category': None, 'support': 0, 'review': True, 'method': 'CNN ResNet-18'}
    # Retirar persona y fondo antes de clasificar; conservar proporción de la prenda.
    crop = rgb[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()
    crop[~mask[ys.min():ys.max()+1, xs.min():xs.max()+1]] = 245
    with torch.inference_mode():
        p = classifier()(torch.from_numpy(tensor_image(Image.fromarray(crop)))[None]).softmax(1)[0].numpy()
    order = np.argsort(p)[::-1]
    best = int(order[0])
    return {'category': CLASSES[best], 'support': round(float(p[best]), 4),
            'review': bool(p[best] < .75 or p[best] - p[order[1]] < .25),
            'method': 'U²-Net + CNN ResNet-18', 'calibrated': False,
            'scores': {c: round(float(v), 4) for c, v in zip(CLASSES, p)}}


@lru_cache(maxsize=2)
def segmenter(filename='u2net_cloth_seg.onnx'):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False
    return ort.InferenceSession(str(MODELS / filename), sess_options=opts,
                               providers=['CPUExecutionProvider'])


def decode_labels(output, size):
    """Las clases discretas se remuestrean con vecino próximo, nunca bilineal."""
    labels = output.argmax(axis=1)[0].astype(np.uint8)
    return np.asarray(Image.fromarray(labels).resize(size, Image.Resampling.NEAREST))


def select_mask(labels, region='auto'):
    """La red puede confundir sus subclases: no asignamos tipo por el índice del píxel.

    Auto utiliza la unión de ropa y la CNN decide el tipo. La selección superior/inferior
    usa posición dentro de la caja detectada cuando hay prendas contiguas en una persona.
    Es un recorte asistido explícito, no una detección de instancias.
    """
    mask = labels > 0
    if region != 'auto' and mask.any():
        ys, _ = np.nonzero(mask)
        cutoff = int(ys.min() + .62 * (ys.max() - ys.min() + 1))
        if region == 'upper':
            mask[cutoff:] = False
        elif region == 'lower':
            mask[:cutoff] = False
        else:
            raise ValueError('Región de prenda inválida')
    return mask


def separate_regions(labels):
    """Divide dos regiones que ocupan alturas distintas en la foto."""
    upper, lower = labels == 1, labels == 2
    min_area = max(32, (labels > 0).sum() * .12)
    if upper.sum() < min_area or lower.sum() < min_area:
        return None
    uy = np.nonzero(upper)[0]
    ly = np.nonzero(lower)[0]
    u0, u1 = np.quantile(uy, [.05, .95])
    l0, l1 = np.quantile(ly, [.05, .95])
    if l0 < u1 - .20 * min(u1 - u0, l1 - l0):
        return None
    return upper, lower


def model_output(rgb, filename, size):
    session = segmenter(filename)
    image = Image.fromarray(rgb).resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, np.float32)
    pixels = (pixels / max(float(pixels.max()), 1e-6) - MEAN) / STD
    return session.run([session.get_outputs()[0].name],
                       {session.get_inputs()[0].name: pixels.transpose(2, 0, 1)[None]})[0]


def clothing_labels(rgb):
    if not status()['ready']:
        raise RuntimeError('La red no está instalada. Revisa requirements-cnn.txt y los pesos del modelo.')
    height, width = rgb.shape[:2]
    return decode_labels(model_output(rgb, 'u2net_cloth_seg.onnx', 768), (width, height))


def segment_result(rgb, labels, mask, region, separated=False, foreground=False):
    from .segmentacion import ResultadoSegmentacion

    enough = mask.sum() >= max(32, mask.size * .005)
    if not enough:
        mask = np.zeros_like(mask)
    classification = classify_crop(rgb, mask)
    if enough:
        from .estampados import type_scores

        visual = type_scores(rgb, mask)
        if separated and region == 'lower':
            # El modelo de ropa encontró una zona inferior distinta de la superior.
            suggested = 'pantalones'
        else:
            suggested = classification['category']
            if visual:
                candidates = ('camisas', 'chaquetas') if separated and region == 'upper' else CLASSES
                first, second = sorted(candidates, key=visual.get, reverse=True)[:2]
                if visual[first] - visual[second] >= .035:
                    suggested = first
                elif foreground and classification['review'] and visual[first] - visual[second] < .035:
                    suggested = None
                classification['visual_scores'] = {label: round(score, 4) for label, score in visual.items()}
        if suggested != classification['category']:
            classification['cnn_category'] = classification['category']
            classification['category'] = suggested
            classification['support'] = None
            classification['review'] = True
    classification['validation'] = status().get('validation', {})
    classification['review'] = bool(classification['review'] or region in ('upper', 'lower'))
    diag = {'method': 'U²-Net ligero' if foreground else 'U²-Net ropa',
            'engine': 'cnn', 'region': region, 'classification': classification,
            'separated_regions': separated,
            'pixel_classes': [int(k) for k in np.unique(labels) if k]}
    return ResultadoSegmentacion(mask, labels, int(labels.max()) + 1, 1, diag)


def analyze(rgb, region='auto'):
    """Selecciona y clasifica una sola prenda."""
    from .segmentacion import componente_mas_grande

    if region not in ('auto', 'upper', 'lower'):
        raise ValueError('Región de prenda inválida')
    labels = clothing_labels(rgb)
    regions = separate_regions(labels)
    if regions is not None:
        if region == 'auto':
            mask = max(regions, key=np.count_nonzero)
        else:
            mask = regions[1 if region == 'lower' else 0]
        kind = 'upper' if mask is regions[0] else 'lower'
        return segment_result(rgb, labels, mask, kind, separated=True)

    foreground = False
    if np.count_nonzero(labels) < max(32, labels.size * .005) and (MODELS / 'u2netp.onnx').is_file():
        scores = model_output(rgb, 'u2netp.onnx', 320)[0, 0]
        height, width = rgb.shape[:2]
        resized = Image.fromarray(scores).resize((width, height), Image.Resampling.BILINEAR)
        labels = (np.asarray(resized) >= .60).astype(np.uint8)
        foreground = True

    mask = componente_mas_grande(select_mask(labels, region))
    return segment_result(rgb, labels, mask, region, foreground=foreground)

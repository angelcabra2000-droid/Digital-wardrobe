"""Reconocimiento local de motivos impresos con CLIP."""
from functools import lru_cache
from pathlib import Path

import logging

import numpy as np
from PIL import Image, ImageOps

MODEL_DIR = Path(__file__).resolve().parent.parent / 'datos' / 'red_neuronal' / 'clip'
PRINTS = {
    'Sin estampado': ('plain solid color clothing with no printed design',
                      'unprinted fabric, including a solid color knit or textured fabric'),
    'Palmeras': ('clothing printed with palm trees or palm fronds',
                 'a palm tree motif on printed fabric'),
    'Flores': ('a floral print of flowers on clothing', 'fabric printed with flowers'),
    'Hojas': ('a botanical print with leaves on clothing', 'fabric printed with foliage'),
    'Animal print': ('leopard, zebra or snakeskin print on clothing', 'animal skin patterned fabric'),
    'Rayas': ('striped clothing with contrasting colored stripes', 'a fabric with a stripe print'),
    'Cuadros': ('plaid, gingham or checked print on clothing', 'fabric with squares in a grid'),
    'Lunares': ('clothing printed with polka dots', 'a polka dot print on fabric'),
    'Gráfico o texto': ('clothing with a large printed illustration, lettering or graphic',
                        'a printed graphic or typography on a garment'),
    'Geométrico': ('a geometric print on clothing', 'fabric with printed abstract geometric shapes'),
}
TYPES = {
    'camisas': ('a button down shirt or dress shirt', 'a blouse or a t-shirt'),
    'chaquetas': ('a jacket or coat worn as outerwear', 'a blazer or bomber jacket'),
    'pantalones': ('a pair of trousers or jeans', 'a pair of shorts or pants'),
}


@lru_cache(maxsize=1)
def _model():
    if not (MODEL_DIR / 'pytorch_model.bin').is_file():
        return None
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(MODEL_DIR, local_files_only=True, use_safetensors=False)
        processor = CLIPProcessor.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=False)
    except (ImportError, OSError, RuntimeError, ValueError):
        logging.getLogger(__name__).exception('No se pudo cargar CLIP')
        return None
    return model.eval(), processor


@lru_cache(maxsize=2)
def _text_vectors(kind):
    import torch
    model, processor = _model()
    groups = PRINTS if kind == 'print' else TYPES
    prompts = [f'a photo of {label}' for labels in groups.values() for label in labels]
    tokens = processor(text=prompts, return_tensors='pt', padding=True)
    with torch.inference_mode():
        vectors = model.get_text_features(**tokens)
        return torch.nn.functional.normalize(vectors).reshape(len(groups), 2, -1).mean(1)


def _scores(image, kind):
    if _model() is None:
        return {}
    import torch
    model, processor = _model()
    with torch.inference_mode():
        vectors = model.get_image_features(**processor(images=image, return_tensors='pt'))
        similarity = (torch.nn.functional.normalize(vectors) @ _text_vectors(kind).T)[0]
    return dict(zip(PRINTS if kind == 'print' else TYPES, similarity.tolist()))


def garment_crop(rgb, mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    bounds = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
    crop = rgb[bounds].copy()
    crop[~mask[bounds]] = 245
    return Image.fromarray(crop)


def classify_print(rgb, mask):
    image = garment_crop(rgb, mask)
    if image is None or min(image.size) < 32:
        return {'label': 'No concluyente', 'estimated': True}
    scores = _scores(ImageOps.pad(image, (224, 224), color=(245, 245, 245)), 'print')
    if not scores:
        return {'label': 'No analizado', 'estimated': True}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, second = ranked[:2]
    label = best if best == 'Sin estampado' or scores[best] - scores[second] >= .009 else 'No concluyente'
    return {'label': label, 'estimated': True, 'method': 'CLIP',
            'scores': {name: round(score, 4) for name, score in scores.items()}}


def type_scores(rgb, mask):
    image = garment_crop(rgb, mask)
    if image is None:
        return {}
    return _scores(ImageOps.pad(image, (224, 224), color=(245, 245, 245)), 'type')

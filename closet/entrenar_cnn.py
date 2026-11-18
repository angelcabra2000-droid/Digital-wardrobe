"""Adaptación reproducible de ResNet-18: solo entrenamiento, validación por grupos.

No consume fotos de usuarios ni las 39 fotografías reservadas para prueba.
El segmentador U²-Net usa pesos externos y no se entrena con este script.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance

from .red_neuronal import ROOT, MODELS, CLASSES, tensor_image
from .pipeline import procesar_foto
from .config import PreprocesoConfig


def split_groups(rows, seed=42):
    rng = random.Random(seed)
    train, valid = [], []
    for category in CLASSES:
        groups = sorted({r['grupo'] or r['id'] for r in rows if r['categoria'] == category})
        rng.shuffle(groups)
        held = set(groups[:max(1, round(.2 * len(groups)))])
        for row in rows:
            if row['categoria'] == category:
                (valid if (row['grupo'] or row['id']) in held else train).append(row)
    if not train or set(r['grupo'] or r['id'] for r in train) & set(r['grupo'] or r['id'] for r in valid):
        raise ValueError('División por grupos inválida')
    return train, valid


def load_rows(root):
    with (root/'datos/dataset.csv').open(encoding='utf-8-sig') as stream:
        return [r for r in csv.DictReader(stream) if r['incluir'].strip().lower() == 'si'
                and r['split'] == 'entrenamiento' and r['categoria'] in CLASSES]


def train(root=ROOT, output=MODELS, epochs=250):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights

    torch.manual_seed(42); np.random.seed(42); random.seed(42); torch.set_num_threads(4)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    rows = load_rows(root)
    development, valid = split_groups(rows)
    base = root/'Aplicacion base de datos'
    encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.eval().to(device)
    cache = root/'resultados/cnn_entrenamiento'
    cache.mkdir(parents=True, exist_ok=True)
    vectors, labels, ids = [], [], []
    clean = {}
    for n, row in enumerate(rows):
        cache_file = cache/(row['id']+'.npz')
        if cache_file.exists():
            raw = np.load(cache_file)['vectors']
        else:
            # Dos vistas: imagen de catálogo y recorte clásico; no pseudoetiquetas de píxel.
            result = procesar_foto(base/row['ruta'], cfg_pre=PreprocesoConfig(lado_max=400))
            original = Image.fromarray(result.preproceso.original)
            images = [original]
            if result.segmentacion.encontrada:
                mask = result.segmentacion.mascara
                ys, xs = np.nonzero(mask)
                crop = result.preproceso.original[ys.min():ys.max()+1,xs.min():xs.max()+1].copy()
                crop[~mask[ys.min():ys.max()+1,xs.min():xs.max()+1]] = 245
                images.append(Image.fromarray(crop))
            augmented=[]
            for im in images:
                augmented.extend([im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                                  ImageEnhance.Color(im).enhance(.35), ImageEnhance.Brightness(im).enhance(.8)])
            x = torch.from_numpy(np.stack([tensor_image(im) for im in augmented])).to(device)
            with torch.inference_mode(): raw = encoder(x).cpu().numpy()
            np.savez_compressed(cache_file,vectors=raw)
        # La primera vista sin aumento es la consulta de validación; no entra en ajuste.
        clean[row['id']] = raw[0]
        vectors.extend(raw); labels.extend([CLASSES.index(row['categoria'])]*len(raw));ids.extend([row['id']]*len(raw))
        print(f'CNN: características {n+1}/{len(rows)} {row["id"]}',flush=True)
    encoder.cpu()
    x=torch.tensor(np.stack(vectors));y=torch.tensor(labels)
    dev_ids={r['id'] for r in development}
    keep=torch.tensor([i in dev_ids for i in ids])
    vx=torch.tensor(np.stack([clean[r['id']] for r in valid]));vy=torch.tensor([CLASSES.index(r['categoria']) for r in valid])
    counts=torch.bincount(y[keep],minlength=3).float()
    weights=counts.sum()/(3*counts)
    head=torch.nn.Linear(512,3)
    initial=copy.deepcopy(head.state_dict())
    optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.05)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weights,label_smoothing=.05)
    best=None
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();loss=loss_fn(head(x[keep]),y[keep]);loss.backward();optimizer.step()
        with torch.no_grad():
            logits=head(vx);accuracy=float((logits.argmax(1)==vy).float().mean());vloss=float(torch.nn.functional.cross_entropy(logits,vy))
        key=(accuracy,-vloss)
        if best is None or key>best[0]:best=(key,epoch,copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[2])
    with torch.no_grad():prediction=head(vx).argmax(1).tolist()
    matrix=np.zeros((3,3),dtype=int)
    for real,pred in zip(vy.tolist(),prediction):matrix[real,pred]+=1
    validation={'method':'reserva interna estratificada por grupos del entrenamiento',
                'total':len(valid),'correct':int(matrix.trace()),'accuracy':round(float(matrix.trace()/matrix.sum()),4),
                'classes':list(CLASSES),'confusion':matrix.tolist(),'test_set_used':False,
                'training_ids':[r['id'] for r in development],'validation_ids':[r['id'] for r in valid],
                'selection_epochs':epochs,'selected_epoch':best[1],
                'note':'Se seleccionó la época con esta validación; no es una prueba independiente.'}
    # Reajuste en las 90 referencias, con el número de épocas seleccionado internamente.
    head.load_state_dict(initial)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.05)
    counts=torch.bincount(y,minlength=3).float()
    loss_fn=torch.nn.CrossEntropyLoss(weight=counts.sum()/(3*counts),label_smoothing=.05)
    for _ in range(best[1]):
        optimizer.zero_grad();loss=loss_fn(head(x),y);loss.backward();optimizer.step()
    model=resnet18(weights=None);model.fc=head
    state=encoder.state_dict();state.update({'fc.weight':head.weight.detach(),'fc.bias':head.bias.detach()})
    model.load_state_dict(state)
    output.mkdir(parents=True,exist_ok=True)
    torch.save({'classes':list(CLASSES),'state_dict':model.state_dict()},output/'clasificador.pt')
    manifest={'version':1,'architecture':'U²-Net ropa + ResNet-18 ImageNet / cabeza lineal adaptada',
              'records':len(rows),'validation':validation,'training_ids':[r['id'] for r in rows],
              'user_photos_used':False,'segmentation_trained_locally':False,
              'segmentation_source':'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx',
              'segmentation_md5':'2434d1f3cb744e0e49386c906e5a08bb',
              'classifier_sha256':hashlib.sha256((output/'clasificador.pt').read_bytes()).hexdigest(),
              'limitations':['Segmentación no validada con máscaras manuales locales.',
                             'No detecta instancias individuales; revisar fotos con varias prendas.',
                             'Solo tres tipos admitidos; valores softmax sin calibración.']}
    (output/'modelo.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(validation,ensure_ascii=False),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=250);args=p.parse_args()
    train(epochs=args.epochs)

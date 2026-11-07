"""Medición de la segmentación contra máscaras dibujadas a mano (`<nombre>_gt.png`).

Métricas por foto: IoU, precisión, recall y F1 (sobre los píxeles de la prenda).
La comparación se hace a la resolución de trabajo (la de `ResultadoPreproceso.original`): la máscara
a mano se reduce con vecino más cercano, así no se inventan píxeles intermedios.

Regla del proyecto: se ajusta con las máscaras de ENTRENAMIENTO; las de PRUEBA se miden una sola vez al final.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PreprocesoConfig, SegmentacionConfig
from .dataset import Registro, nombre_en_muestra
from .segmentacion import ResultadoSegmentacion, segmentar_archivo


@dataclass
class Par:
    registro: Registro
    foto: Path
    mascara: Path


@dataclass
class Medicion:
    par: Par
    iou: float
    precision: float
    recall: float
    f1: float
    area_pred: float   # fracción de la imagen
    area_gt: float
    encontrada: bool


def metricas(pred: np.ndarray | None, gt: np.ndarray) -> tuple[float, float, float, float]:
    """(IoU, precisión, recall, F1). Si no hay predicción, todo es 0."""
    if pred is None or not pred.any():
        return 0.0, 0.0, 0.0, 0.0
    inter = float(np.logical_and(pred, gt).sum())
    p, g = float(pred.sum()), float(gt.sum())
    union = p + g - inter
    iou = inter / union if union else 0.0
    precision = inter / p if p else 0.0
    recall = inter / g if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return iou, precision, recall, f1


def reducir_mascara(gt: np.ndarray, alto: int, ancho: int) -> np.ndarray:
    """Cambia el tamaño de una máscara booleana con vecino más cercano."""
    if gt.shape == (alto, ancho):
        return gt
    ys = np.minimum((np.arange(alto) + 0.5) * gt.shape[0] / alto, gt.shape[0] - 1).astype(int)
    xs = np.minimum((np.arange(ancho) + 0.5) * gt.shape[1] / ancho, gt.shape[1] - 1).astype(int)
    return gt[ys][:, xs]


def cargar_mascara(ruta: str | Path) -> np.ndarray:
    return np.asarray(Image.open(ruta).convert("L")) > 127


def buscar_pares(registros: list[Registro], carpeta: str | Path, conjunto: str) -> tuple[list[Par], list[str]]:
    """Fotos de la muestra del `conjunto` que tienen foto y máscara en `carpeta`. Devuelve (pares, faltantes)."""
    carpeta = Path(carpeta)
    pares, faltan = [], []
    for r in registros:
        if not (r.muestra_mascara and r.incluir and r.split == conjunto):
            continue
        nombre = nombre_en_muestra(r)
        foto, mascara = carpeta / nombre, carpeta / (Path(nombre).stem + "_gt.png")
        if foto.exists() and mascara.exists():
            pares.append(Par(r, foto, mascara))
        else:
            faltan.append(nombre)
    return pares, faltan


def medir_par(par: Par, cfg_pre: PreprocesoConfig, cfg: SegmentacionConfig,
              preprocesar: bool = True) -> tuple[Medicion, ResultadoSegmentacion, np.ndarray, np.ndarray]:
    pre, res = segmentar_archivo(par.foto, cfg_pre, cfg, preprocesar)
    h, w, _ = pre.original.shape
    gt = reducir_mascara(cargar_mascara(par.mascara), h, w)
    iou, pr, rc, f1 = metricas(res.mascara, gt)
    m = Medicion(par, iou, pr, rc, f1, float(res.mascara.mean()) if res.encontrada else 0.0,
                 float(gt.mean()), res.encontrada)
    return m, res, pre.original, gt


def evaluar(pares: list[Par], cfg_pre: PreprocesoConfig | None = None, cfg: SegmentacionConfig | None = None,
            preprocesar: bool = True, al_terminar_foto=None) -> list[Medicion]:
    cfg_pre, cfg = cfg_pre or PreprocesoConfig(), cfg or SegmentacionConfig()
    out = []
    for par in pares:
        m, res, rgb, gt = medir_par(par, cfg_pre, cfg, preprocesar)
        out.append(m)
        if al_terminar_foto:
            al_terminar_foto(m, res, rgb, gt)
    return out


def resumen(mediciones: list[Medicion]) -> dict[str, dict[str, float]]:
    """Promedios por categoría y total. Incluye cuántas fotos quedaron bajo IoU 0.8 (las 'fallas')."""
    grupos: dict[str, list[Medicion]] = {}
    for m in mediciones:
        grupos.setdefault(m.par.registro.categoria, []).append(m)
        grupos.setdefault("TOTAL", []).append(m)
    out = {}
    for cat, ms in sorted(grupos.items(), key=lambda kv: (kv[0] == "TOTAL", kv[0])):
        a = lambda f: float(np.mean([getattr(m, f) for m in ms]))
        out[cat] = {"n": len(ms), "iou": a("iou"), "iou_min": min(m.iou for m in ms),
                    "precision": a("precision"), "recall": a("recall"), "f1": a("f1"),
                    "bajo_080": sum(m.iou < 0.8 for m in ms)}
    return out


def superposicion(rgb: np.ndarray, pred: np.ndarray | None, gt: np.ndarray) -> np.ndarray:
    """Imagen para revisar a ojo: verde = acierto, rojo = faltó (solo a mano), azul = sobró (solo predicción)."""
    pred = np.zeros_like(gt) if pred is None else pred
    out = (rgb * 0.45).astype(np.uint8)
    for sel, color in ((pred & gt, (60, 200, 60)), (gt & ~pred, (230, 40, 40)), (pred & ~gt, (50, 90, 240))):
        out[sel] = (0.45 * out[sel] + 0.55 * np.array(color)).astype(np.uint8)
    return out

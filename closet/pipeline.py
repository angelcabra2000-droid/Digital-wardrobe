"""Encadena los pasos 1 a 3 para una foto: preprocesar -> segmentar -> revisar -> normalizar."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .alertas import revisar_segmentacion
from .config import AlertasConfig, NormalizacionConfig, PreprocesoConfig, SegmentacionConfig
from .imagen_io import cargar_imagen
from .normalizacion import ResultadoNormalizacion, normalizar_prenda
from .preprocesamiento import ResultadoPreproceso, preprocesar_imagen
from .segmentacion import ResultadoSegmentacion, segmentar_imagen


@dataclass
class ResultadoPrenda:
    ruta: Path
    preproceso: ResultadoPreproceso
    segmentacion: ResultadoSegmentacion
    normalizada: ResultadoNormalizacion | None   # None si no se encontró prenda
    alertas: list[str] = field(default_factory=list)
    metricas: dict = field(default_factory=dict)


def procesar_foto(ruta, cfg_pre: PreprocesoConfig | None = None, cfg_seg: SegmentacionConfig | None = None,
                  cfg_norm: NormalizacionConfig | None = None, cfg_alertas: AlertasConfig | None = None,
                  preprocesar: bool = True, motor: str = 'clasico', region: str = 'auto') -> ResultadoPrenda:
    """Todo el recorrido de una foto. Segmenta sobre la imagen normalizada y mide el color sobre la original."""
    pre = preprocesar_imagen(cargar_imagen(ruta), cfg_pre)
    if motor == 'cnn':
        from .red_neuronal import analyze
        seg = analyze(pre.original, region)
    elif motor == 'clasico':
        seg = segmentar_imagen(pre.normalizada if preprocesar else pre.original, cfg_seg)
    else:
        raise ValueError('Motor de segmentación inválido')
    return _finalizar_prenda(ruta, pre, seg, cfg_norm, cfg_alertas)


def _finalizar_prenda(ruta, pre, seg, cfg_norm=None, cfg_alertas=None):
    if not seg.encontrada:
        return ResultadoPrenda(Path(ruta), pre, seg, None, ["sin_prenda"])
    alertas, metricas = revisar_segmentacion(pre.original, seg.mascara, cfg_alertas)
    norm = normalizar_prenda(pre.original, seg.mascara, cfg_norm)
    return ResultadoPrenda(Path(ruta), pre, seg, norm, alertas, {**metricas, **norm.meta})

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
from .segmentacion import ResultadoSegmentacion, componente_mas_grande, dilatar_octogono, segmentar_imagen


@dataclass
class ResultadoPrenda:
    ruta: Path
    preproceso: ResultadoPreproceso
    segmentacion: ResultadoSegmentacion
    normalizada: ResultadoNormalizacion | None   # None si no se encontró prenda
    alertas: list[str] = field(default_factory=list)
    metricas: dict = field(default_factory=dict)
    mascaras: dict[str, np.ndarray] = field(default_factory=dict)


def fusionar_mascaras(neural, clasica):
    """K-means corrige una banda estrecha cuando ambos recortes coinciden."""
    if not neural.any():
        return clasica, 0., 'clasico_sin_mascara_neuronal'
    agreement = float((neural & clasica).sum() / (neural | clasica).sum())
    if agreement < .45:
        return neural, agreement, 'u2net_desacuerdo'
    radius = max(1, round(.008 * max(neural.shape)))
    core = neural & ~dilatar_octogono(~neural, radius)
    band = dilatar_octogono(neural, radius)
    fused = componente_mas_grande(core | (band & clasica))
    if abs(int(fused.sum()) - int(neural.sum())) > .15 * neural.sum():
        return neural, agreement, 'u2net_cambio_excesivo'
    return fused, agreement, 'hibrido'


def procesar_foto(ruta, cfg_pre: PreprocesoConfig | None = None, cfg_seg: SegmentacionConfig | None = None,
                  cfg_norm: NormalizacionConfig | None = None, cfg_alertas: AlertasConfig | None = None,
                  preprocesar: bool = True, motor: str = 'clasico', region: str = 'auto',
                  rotacion: int = 0) -> ResultadoPrenda:
    """Todo el recorrido de una foto. Segmenta sobre la imagen normalizada y mide el color sobre la original."""
    if rotacion not in (0, 90, 180, 270):
        raise ValueError('Rotación inválida')
    rgb = cargar_imagen(ruta)
    pre = preprocesar_imagen(np.ascontiguousarray(np.rot90(rgb, -(rotacion // 90))), cfg_pre)
    pre.pasos['rotacion_manual_grados'] = rotacion
    masks = {}
    base = pre.normalizada if preprocesar else pre.original
    if motor == 'cnn':
        from .red_neuronal import analyze
        seg = analyze(pre.original, region)
    elif motor == 'clasico':
        seg = segmentar_imagen(base, cfg_seg)
    elif motor in ('hibrido', 'u2net'):
        from .red_neuronal import clothing_labels, select_mask, separate_regions

        if region not in ('auto', 'upper', 'lower'):
            raise ValueError('Región de prenda inválida')
        labels = clothing_labels(pre.original)
        regions = separate_regions(labels)
        if regions is not None:
            selected = max(regions, key=np.count_nonzero) if region == 'auto' else regions[int(region == 'lower')]
        else:
            selected = select_mask(labels, region)
        neural = componente_mas_grande(selected)
        if neural.sum() < max(32, neural.size * .005):
            neural = np.zeros_like(neural)
        masks['mascara_u2net'] = neural
        if motor == 'u2net':
            seg = ResultadoSegmentacion(neural, labels, int(labels.max()) + 1, 1, {'engine':'u2net'})
        else:
            seg = segmentar_imagen(base, cfg_seg)
            classical = seg.mascara if seg.encontrada else np.zeros_like(neural)
            masks['mascara_kmeans'] = classical
            seg.mascara, overlap, strategy = fusionar_mascaras(neural, classical)
            seg.diagnostico.update(engine='hibrido', estrategia=strategy, iou_entre_mascaras=round(overlap, 3))
    else:
        raise ValueError('Motor de segmentación inválido')
    result = _finalizar_prenda(ruta, pre, seg, cfg_norm, cfg_alertas)
    result.mascaras = masks
    return result


def _finalizar_prenda(ruta, pre, seg, cfg_norm=None, cfg_alertas=None):
    if not seg.encontrada:
        return ResultadoPrenda(Path(ruta), pre, seg, None, ["sin_prenda"])
    alertas, metricas = revisar_segmentacion(pre.original, seg.mascara, cfg_alertas)
    norm = normalizar_prenda(pre.original, seg.mascara, cfg_norm)
    return ResultadoPrenda(Path(ruta), pre, seg, norm, alertas, {**metricas, **norm.meta})

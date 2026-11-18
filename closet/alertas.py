"""Alertas de segmentación sospechosa: avisan cuando el recorte probablemente está mal.

No descartan nada por sí solas (igual que las advertencias de calidad): la app puede mostrar
"revisa esta foto" o excluir la prenda del closet automático. Umbrales en `AlertasConfig`.

Alertas:
  prenda_muy_grande      la máscara ocupa casi toda la foto (se pegó fondo)
  prenda_muy_chica       la máscara es diminuta (se perdió casi toda la prenda)
  prenda_parecida_fondo  el color medio de la prenda y del fondo es casi el mismo
  fondo_complejo         el fondo varía mucho de color (piso, cama o tela estampada)
"""
from __future__ import annotations

import numpy as np

from .config import AlertasConfig
from .segmentacion import dilatar_octogono, rgb_a_lab


def fondo_impide_segmentar(alertas):
    """Bloquea recortes con indicios de fondo adherido o prenda incompleta."""
    deficient = 'prenda_parecida_fondo' in alertas
    return (('fondo_complejo' in alertas and
             (deficient or 'prenda_muy_grande' in alertas or 'prenda_muy_chica' in alertas))
            or ('borde_extenso' in alertas and deficient)
            or 'fondo_irregular_en_borde' in alertas
            or 'varios_bordes' in alertas)


def revisar_segmentacion(original: np.ndarray, mascara: np.ndarray,
                         cfg: AlertasConfig | None = None) -> tuple[list[str], dict]:
    """Devuelve (lista de alertas, métricas usadas). `original` es la imagen sin corregir, del tamaño de la máscara."""
    cfg = cfg or AlertasConfig()
    h, w = mascara.shape
    L = max(h, w)
    area = float(mascara.mean())
    metricas = {"area": round(area, 4), "contraste": None, "fondo_sigma": None}
    alertas = []
    border = {name: round(float(edge.mean()), 3) for name, edge in
              zip(('superior', 'inferior', 'izquierdo', 'derecho'),
                  (mascara[0], mascara[-1], mascara[:, 0], mascara[:, -1]))}
    metricas['contacto_bordes'] = border
    contacts = sum(value >= .18 for value in border.values())
    if contacts >= 2:
        alertas.append("varios_bordes")
    elif contacts:
        alertas.append("borde_extenso")
    if area > cfg.area_max:
        alertas.append("prenda_muy_grande")
    if area < cfg.area_min:
        alertas.append("prenda_muy_chica")

    # Fondo = lo que queda lejos de la prenda (así el borde, mezclado, no cuenta); prenda = su parte interior
    fondo = ~dilatar_octogono(mascara, max(1, int(round(cfg.separacion * L))))
    interior = mascara & ~dilatar_octogono(~mascara, max(1, int(round(0.01 * L))))
    if fondo.sum() >= 100 and interior.sum() >= 100:
        lab = rgb_a_lab(original.reshape(-1, 3).astype(np.float64)).reshape(h, w, 3)
        contraste = float(np.linalg.norm(lab[interior].mean(axis=0) - lab[fondo].mean(axis=0)))
        sigma = float(np.sqrt(lab[fondo].var(axis=0).sum()))
        metricas["contraste"], metricas["fondo_sigma"] = round(contraste, 1), round(sigma, 1)
        if contraste < cfg.contraste_min:
            alertas.append("prenda_parecida_fondo")
        if sigma > cfg.fondo_sigma_max:
            alertas.append("fondo_complejo")
        if contacts == 1 and sigma >= .8 * cfg.fondo_sigma_max:
            alertas.append("fondo_irregular_en_borde")
    return alertas, metricas

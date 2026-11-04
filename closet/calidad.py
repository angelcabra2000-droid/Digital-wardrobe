"""Control de calidad de la foto: detecta imágenes borrosas, oscuras, sin contraste, etc.

Solo emite ADVERTENCIAS; nunca descarta una foto por sí solo. Las métricas se calculan sobre la
imagen ya redimensionada (misma escala para todas) y antes de corregir color o suavizar.
"""
from __future__ import annotations

import numpy as np

from .config import CalidadConfig

PESOS_LUMINANCIA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luminancia(img: np.ndarray) -> np.ndarray:
    """Imagen RGB uint8 -> luminancia float32 en [0, 255]."""
    return img.astype(np.float32) @ PESOS_LUMINANCIA


def varianza_laplaciano(lum: np.ndarray) -> float:
    """Medida de nitidez: varianza del Laplaciano (núcleo [[0,1,0],[1,-4,1],[0,1,0]]).

    Los bordes nítidos producen respuestas fuertes; una foto borrosa deja el Laplaciano casi plano.
    """
    lap = (lum[:-2, 1:-1] + lum[2:, 1:-1] + lum[1:-1, :-2] + lum[1:-1, 2:]
           - 4.0 * lum[1:-1, 1:-1])
    return float(lap.var())


def evaluar_calidad(img: np.ndarray, cfg: CalidadConfig | None = None) -> dict:
    """Devuelve las métricas de la foto y la lista de advertencias."""
    cfg = cfg or CalidadConfig()
    lum = luminancia(img)
    alto, ancho = img.shape[:2]

    metricas = {
        "alto": alto,
        "ancho": ancho,
        "nitidez": varianza_laplaciano(lum),
        "brillo": float(lum.mean()),
        "contraste": float(lum.std()),
        "frac_saturada": float((lum >= 250).mean()),
        "frac_oscura": float((lum <= 5).mean()),
    }

    advertencias = []
    if min(alto, ancho) < cfg.lado_min:
        advertencias.append("baja_resolucion")
    if metricas["nitidez"] < cfg.nitidez_min:
        advertencias.append("borrosa")
    if metricas["brillo"] < cfg.brillo_min:
        advertencias.append("oscura")
    if metricas["frac_saturada"] > cfg.frac_saturada_max:
        advertencias.append("sobreexpuesta")
    if metricas["contraste"] < cfg.contraste_min:
        advertencias.append("bajo_contraste")

    metricas["advertencias"] = advertencias
    return metricas

"""Paso 2b: refinar el borde de la máscara de K-means con el umbral de Otsu (solo NumPy).

K-means da una máscara casi buena pero con el borde "gordo" o "flaco" unos píxeles (halos, sombras suaves, borde
mezclado con el fondo). Aquí solo se vuelve a decidir ese borde:

  1) BANDA        dilatación menos erosión de la máscara de K-means. Lo de adentro de la banda es prenda segura y
                  lo de afuera es fondo seguro; no se toca.
  2) REFERENCIAS  colores (Lab) de la prenda segura y del fondo seguro cercano, resumidos con K-means (hasta 3 colores
                  cada uno, así una camisa a rayas o un fondo con dos tonos no confunden al umbral).
  3) PUNTAJE      por píxel de la banda: (distancia al color de prenda más parecido) - (distancia al color de fondo
                  más parecido). Negativo = se parece más a la prenda.
  4) OTSU         umbral que maximiza la varianza entre clases sobre el puntaje de la banda. Solo se aplica en la banda.
  5) LIMPIEZA     se queda con el componente más grande y se alisa el borde con un filtro de moda 3x3.

Salvaguardas (si alguna falla, se devuelve la máscara de K-means sin cambios y el motivo queda en el diagnóstico):
  - la separación de Otsu (varianza entre clases / varianza total) debe ser >= `separacion_min`: si prenda y fondo
    se parecen, el histograma no es bimodal y Otsu sería ruido;
  - el refinamiento no puede cambiar más de `cambio_max` del área de la máscara.
"""
from __future__ import annotations

import numpy as np

from .config import SegmentacionConfig
from .segmentacion import (componente_mas_grande, dilatacion, erosion, filtro_caja, kmeans, rgb_a_lab)

_MAX_MUESTRA = 3000   # píxeles que se usan para resumir los colores de referencia


def umbral_otsu(valores: np.ndarray, bins: int = 256) -> tuple[float, float]:
    """Umbral de Otsu de un conjunto de valores. Devuelve (umbral, separación en 0-1).

    La separación es varianza_entre_clases / varianza_total: ~0 si los valores forman un solo grupo,
    ~1 si son dos grupos muy bien separados.
    """
    v = np.asarray(valores, dtype=np.float64).ravel()
    if v.size < 2 or v.max() == v.min():
        return float(v[0]) if v.size else 0.0, 0.0
    hist, bordes = np.histogram(v, bins=bins)
    centros = (bordes[:-1] + bordes[1:]) / 2
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * centros)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        entre = (mu_t * w0 - mu) ** 2 / (w0 * w1)
    entre = np.nan_to_num(entre, nan=0.0, posinf=0.0, neginf=0.0)
    # Si hay un hueco vacío entre los dos grupos, muchos umbrales empatan: se toma el centro de ese tramo
    i = int(np.median(np.flatnonzero(entre >= entre.max() * (1 - 1e-9))))
    total = float(((centros - mu_t) ** 2 * p).sum())
    separacion = float(entre[i] / total) if total > 0 else 0.0
    return float(centros[i]), min(separacion, 1.0)


def _centros_color(lab: np.ndarray, k: int, semilla: int) -> np.ndarray:
    """Hasta `k` colores representativos de un conjunto de píxeles Lab (K-means sobre una muestra)."""
    if len(lab) > _MAX_MUESTRA:
        idx = np.random.default_rng(semilla).choice(len(lab), _MAX_MUESTRA, replace=False)
        lab = lab[idx]
    k = max(1, min(k, len(lab)))
    _, C, _ = kmeans(lab, k, seed=semilla)
    return C


def _distancia_min(lab: np.ndarray, centros: np.ndarray) -> np.ndarray:
    d2 = ((lab[:, None, :] - centros[None]) ** 2).sum(2)
    return np.sqrt(d2.min(axis=1))


def refinar_con_otsu(rgb: np.ndarray, mascara: np.ndarray, cfg: SegmentacionConfig | None = None
                     ) -> tuple[np.ndarray, dict]:
    """Refina el borde de `mascara` (bool) usando Otsu. Devuelve (máscara_refinada, diagnóstico)."""
    cfg = cfg or SegmentacionConfig()
    h, w = mascara.shape
    diag: dict = {"aplicado": False, "motivo": ""}
    r = max(2, int(round(cfg.banda_otsu * max(h, w))))
    diag["banda_px"] = r

    seguro_prenda = erosion(mascara, r)
    dentro = dilatacion(mascara, r)
    banda = dentro & ~seguro_prenda
    anillo = dilatacion(mascara, 3 * r) & ~dentro
    seguro_fondo = anillo if anillo.sum() >= 50 else ~dentro
    if seguro_prenda.sum() < 50 or seguro_fondo.sum() < 50 or banda.sum() < 50:
        diag["motivo"] = "sin referencias suficientes (prenda muy fina o sin fondo alrededor)"
        return mascara, diag

    lab = rgb_a_lab(rgb.reshape(-1, 3).astype(np.float64)).reshape(h, w, 3)
    c_prenda = _centros_color(lab[seguro_prenda], 3, cfg.semilla)
    c_fondo = _centros_color(lab[seguro_fondo], 3, cfg.semilla)

    lab_banda = lab[banda]
    puntaje = _distancia_min(lab_banda, c_prenda) - _distancia_min(lab_banda, c_fondo)
    umbral, separacion = umbral_otsu(puntaje)
    diag.update(umbral=umbral, separacion=separacion)
    if separacion < cfg.separacion_otsu_min:
        diag["motivo"] = "prenda y fondo muy parecidos en el borde (Otsu poco confiable)"
        return mascara, diag

    nueva = seguro_prenda.copy()
    nueva[banda] = puntaje < umbral
    nueva = filtro_caja(nueva.astype(np.float64), 1) > 0.5
    nueva = componente_mas_grande(nueva) if nueva.any() else nueva

    area = max(int(mascara.sum()), 1)
    cambio = int((nueva ^ mascara).sum()) / area
    diag["cambio"] = cambio
    if cambio > cfg.cambio_max_otsu:
        diag["motivo"] = "el refinamiento cambiaba demasiado la máscara"
        return mascara, diag

    diag["aplicado"] = True
    return nueva, diag

"""Paso 3: normalización geométrica de la prenda.

Deja todas las prendas en el mismo formato para que las características del paso 4 sean comparables:
  1. momentos de la máscara  -> centro y eje principal
  2. enderezar               -> si la prenda es claramente alargada y está inclinada, el eje principal pasa a vertical
  3. recortar + escalar      -> la caja de la prenda ocupa un lienzo fijo (lado x lado), centrada, sin deformarla

Todo se hace con UNA sola transformación afín y remuestreo bilineal escrito a mano (un solo remuestreo = menos
borrosidad). El color se toma de la imagen ORIGINAL (sin corregir), no de la normalizada.

Limitaciones deliberadas:
  - Nunca se gira más de `giro_max` (45°): un eje principal horizontal es ambiguo (camisas con mangas abiertas),
    así que esas prendas se dejan como están.
  - No se resuelve el "boca abajo" (180°): se asume que la foto viene con la prenda más o menos derecha.
  - La proporción alto/ancho se guarda en `meta`, porque al escalar a un lienzo cuadrado ya no se ve en la imagen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import NormalizacionConfig
from .preprocesamiento import suavizar_gaussiano
from .segmentacion import erosion


@dataclass
class ResultadoNormalizacion:
    rgb: np.ndarray        # (lado, lado, 3) uint8, color ORIGINAL de la prenda; 0 fuera de la máscara
    mascara: np.ndarray    # (lado, lado) bool
    nucleo: np.ndarray     # (lado, lado) bool: máscara erosionada, sin el borde contaminado por el fondo
    meta: dict = field(default_factory=dict)


def momentos(mascara: np.ndarray) -> dict:
    """Centroide, ángulo del eje mayor y elongación a partir de los momentos centrales de la máscara."""
    ys, xs = np.nonzero(mascara)
    cy, cx = ys.mean(), xs.mean()
    dx, dy = xs - cx, ys - cy
    mu20, mu02, mu11 = (dx * dx).mean(), (dy * dy).mean(), (dx * dy).mean()
    # eigenvalores de la matriz de covarianza [[mu20, mu11], [mu11, mu02]]
    media, dif = (mu20 + mu02) / 2, np.hypot((mu20 - mu02) / 2, mu11)
    l1, l2 = media + dif, max(media - dif, 1e-9)
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)      # ángulo del eje mayor respecto a +x (y hacia abajo)
    return {"cy": float(cy), "cx": float(cx), "theta": float(theta), "elongacion": float(np.sqrt(l1 / l2)),
            "area": int(len(xs))}


def angulo_de_giro(m: dict, cfg: NormalizacionConfig) -> float:
    """Ángulo (radianes) del marco nuevo respecto al original; 0 si no conviene enderezar."""
    if not cfg.enderezar or m["elongacion"] < cfg.elongacion_min:
        return 0.0
    t = m["theta"] - np.pi / 2                            # lleva el eje mayor a la vertical
    if t > np.pi / 2:
        t -= np.pi
    elif t <= -np.pi / 2:
        t += np.pi
    grados = abs(np.degrees(t))
    return float(t) if cfg.giro_min <= grados <= cfg.giro_max else 0.0


def muestrear_bilineal(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Valor de `img` (alto, ancho[, canales]) en las coordenadas reales (x, y); fuera de la imagen vale 0."""
    alto, ancho = img.shape[:2]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    wx, wy = x - x0, y - y0
    f = img.astype(np.float32)
    if f.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]

    def valor(yy, xx):
        dentro = (xx >= 0) & (xx < ancho) & (yy >= 0) & (yy < alto)
        v = f[np.clip(yy, 0, alto - 1), np.clip(xx, 0, ancho - 1)]
        return v * (dentro[..., None] if f.ndim == 3 else dentro)

    return (valor(y0, x0) * (1 - wx) * (1 - wy) + valor(y0, x0 + 1) * wx * (1 - wy)
            + valor(y0 + 1, x0) * (1 - wx) * wy + valor(y0 + 1, x0 + 1) * wx * wy)


def normalizar_prenda(original: np.ndarray, mascara: np.ndarray,
                      cfg: NormalizacionConfig | None = None) -> ResultadoNormalizacion:
    """Recorta, endereza y escala la prenda a un lienzo fijo. `original` debe ser la imagen sin corregir."""
    cfg = cfg or NormalizacionConfig()
    if not mascara.any():
        raise ValueError("la máscara está vacía")
    m = momentos(mascara)
    t = angulo_de_giro(m, cfg)
    e1, e2 = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])   # ejes del marco nuevo

    # Caja de la prenda en el marco nuevo (u: derecha, v: abajo), relativa al centroide
    ys, xs = np.nonzero(mascara)
    px, py = xs - m["cx"], ys - m["cy"]
    u, v = px * e1[0] + py * e1[1], px * e2[0] + py * e2[1]
    umin, umax, vmin, vmax = u.min() - 0.5, u.max() + 0.5, v.min() - 0.5, v.max() + 0.5
    ancho_c, alto_c = umax - umin, vmax - vmin
    L = cfg.lado
    esc = L * (1 - 2 * cfg.margen) / max(ancho_c, alto_c)

    # Para cada píxel del lienzo, de dónde se toma en la imagen original (transformación inversa)
    X, Y = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5)
    uu = (X - L / 2) / esc + (umin + umax) / 2
    vv = (Y - L / 2) / esc + (vmin + vmax) / 2
    sx = m["cx"] + uu * e1[0] + vv * e2[0] - 0.5
    sy = m["cy"] + uu * e1[1] + vv * e2[1] - 0.5

    # Si se reduce mucho, se suaviza antes para no generar aliasing
    sigma = max(0.0, 0.5 / esc - 0.5)
    rgb_src = suavizar_gaussiano(original, sigma) if sigma > 0.3 else original
    m_src = suavizar_gaussiano((mascara * 255).astype(np.uint8), sigma) if sigma > 0.3 else mascara * 255
    rgb = np.rint(np.clip(muestrear_bilineal(rgb_src, sx, sy), 0, 255)).astype(np.uint8)
    masc = muestrear_bilineal(m_src, sx, sy) > 127.5
    rgb[~masc] = 0
    nucleo = erosion(masc, max(1, int(round(cfg.nucleo * L))))

    alto_img, ancho_img = mascara.shape
    meta = {
        "giro_grados": round(float(np.degrees(t)), 1),
        "elongacion": round(m["elongacion"], 3),
        "escala": round(float(esc), 4),
        "ratio_aspecto": round(float(alto_c / ancho_c), 4),         # alto / ancho de la prenda ya enderezada
        "area_relativa": round(m["area"] / (alto_img * ancho_img), 4),
        "centro_x": round(m["cx"] / ancho_img, 3), "centro_y": round(m["cy"] / alto_img, 3),
    }
    return ResultadoNormalizacion(rgb, masc, nucleo, meta)

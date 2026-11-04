"""Preprocesamiento: deja todas las fotos en una forma comparable antes de segmentarlas.

Orden de los pasos (cada uno es una función independiente y probada por separado):
    1. redimensionar_lado_max       -> mismo tamaño de trabajo para todas
    2. evaluar_calidad              -> advertencias (en calidad.py)
    3. balance_blancos_mundo_gris   -> quita tintes de color de la luz
    4. suavizar_gaussiano           -> reduce el ruido del sensor
    5. estirar_contraste            -> unifica la exposición

IMPORTANTE: la imagen normalizada es para SEGMENTAR. El color real de la prenda se debe medir
sobre `original` (redimensionada pero sin corregir), usando la máscara obtenida en la segmentación.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .calidad import evaluar_calidad, luminancia
from .config import CalidadConfig, PreprocesoConfig


# ------------------------------------------------------------------ tamaño
def promediar_bloques(img: np.ndarray, f: int) -> np.ndarray:
    """Reduce la imagen un factor entero `f` promediando bloques f x f (evita el aliasing)."""
    alto, ancho, canales = img.shape
    hh, ww = (alto // f) * f, (ancho // f) * f
    bloques = img[:hh, :ww].astype(np.float32).reshape(hh // f, f, ww // f, f, canales)
    return np.rint(bloques.mean(axis=(1, 3))).astype(np.uint8)


def remuestrear_bilineal(img: np.ndarray, nuevo_alto: int, nuevo_ancho: int) -> np.ndarray:
    """Cambia el tamaño con interpolación bilineal (convención de centro de píxel)."""
    alto, ancho = img.shape[:2]
    ys = np.clip((np.arange(nuevo_alto) + 0.5) * alto / nuevo_alto - 0.5, 0, alto - 1)
    xs = np.clip((np.arange(nuevo_ancho) + 0.5) * ancho / nuevo_ancho - 0.5, 0, ancho - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, alto - 1), np.minimum(x0 + 1, ancho - 1)
    wy = (ys - y0).astype(np.float32)[:, None, None]
    wx = (xs - x0).astype(np.float32)[None, :, None]

    f = img.astype(np.float32)
    arriba = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    abajo = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return np.rint(arriba * (1 - wy) + abajo * wy).astype(np.uint8)


def redimensionar_lado_max(img: np.ndarray, lado_max: int) -> np.ndarray:
    """Reduce la imagen para que su lado mayor mida `lado_max` px, conservando la proporción.

    Si ya es más pequeña, se devuelve igual (nunca se amplía: ampliar solo inventa píxeles).
    """
    alto, ancho = img.shape[:2]
    if max(alto, ancho) <= lado_max:
        return img
    esc = lado_max / max(alto, ancho)
    nuevo_alto, nuevo_ancho = max(1, round(alto * esc)), max(1, round(ancho * esc))
    f = min(alto // nuevo_alto, ancho // nuevo_ancho)
    if f >= 2:  # primero un promedio por bloques, luego el ajuste fino bilineal
        img = promediar_bloques(img, f)
    return remuestrear_bilineal(img, nuevo_alto, nuevo_ancho)


# ------------------------------------------------------------------- color
def balance_blancos_mundo_gris(img: np.ndarray, ganancia_max: float = 1.15,
                               rango_util: tuple[float, float] = (20.0, 235.0)):
    """Balance de blancos por la hipótesis del "mundo gris".

    Supone que el promedio de la escena es gris: si el canal rojo promedia más que el azul,
    la luz es cálida y se compensa. Se ignoran los píxeles casi negros o casi blancos.
    Las ganancias se limitan a [1/ganancia_max, ganancia_max] porque la hipótesis falla cuando
    una prenda de un solo color domina la foto (un pantalón marrón se "corregiría" a gris).

    Devuelve (imagen corregida, ganancias RGB).
    """
    f = img.astype(np.float32)
    lum = luminancia(img)
    util = (lum > rango_util[0]) & (lum < rango_util[1])
    if util.mean() < 0.05:  # casi todo negro/blanco: no hay base para estimar
        return img.copy(), (1.0, 1.0, 1.0)
    medias = f[util].mean(axis=0)
    ganancias = np.clip(medias.mean() / np.maximum(medias, 1e-6), 1.0 / ganancia_max, ganancia_max)
    corregida = np.clip(f * ganancias, 0, 255)
    return np.rint(corregida).astype(np.uint8), tuple(float(g) for g in ganancias)


def estirar_contraste(img: np.ndarray, p_bajo: float = 1.0, p_alto: float = 99.0,
                      expansion_max: float = 1.6):
    """Estira linealmente la luminancia: los percentiles [p_bajo, p_alto] pasan a ocupar 0-255.

    El mismo factor se aplica a los tres canales para no alterar el tono. Se centra el rango en
    el gris medio, lo que también compensa fotos sub o sobreexpuestas. `expansion_max` evita
    amplificar de más una foto que ya tiene poco rango.

    Devuelve (imagen corregida, (luminancia_baja, luminancia_alta)).
    """
    lum = luminancia(img)
    bajo, alto = np.percentile(lum, [p_bajo, p_alto])
    rango = float(alto - bajo)
    if rango < 1.0:  # imagen plana: no hay nada que estirar
        return img.copy(), (float(bajo), float(alto))
    factor = min(255.0 / rango, expansion_max)
    centro = (bajo + alto) / 2.0
    salida = (img.astype(np.float32) - centro) * factor + 127.5
    return np.rint(np.clip(salida, 0, 255)).astype(np.uint8), (float(bajo), float(alto))


# ---------------------------------------------------------------- suavizado
def nucleo_gaussiano(sigma: float) -> np.ndarray:
    """Núcleo gaussiano 1D normalizado (radio = 3 sigma)."""
    radio = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radio, radio + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _convolucion_1d(f: np.ndarray, k: np.ndarray, eje: int) -> np.ndarray:
    radio = len(k) // 2
    relleno = [(0, 0)] * f.ndim
    relleno[eje] = (radio, radio)
    p = np.pad(f, relleno, mode="reflect")
    n = f.shape[eje]
    salida = np.zeros_like(f)
    for i, peso in enumerate(k):
        corte = [slice(None)] * f.ndim
        corte[eje] = slice(i, i + n)
        salida += peso * p[tuple(corte)]
    return salida


def suavizar_gaussiano(img: np.ndarray, sigma: float) -> np.ndarray:
    """Desenfoque gaussiano separable (primero filas, luego columnas)."""
    if sigma <= 0:
        return img.copy()
    k = nucleo_gaussiano(sigma)
    f = _convolucion_1d(img.astype(np.float32), k, eje=0)
    f = _convolucion_1d(f, k, eje=1)
    return np.rint(np.clip(f, 0, 255)).astype(np.uint8)


# ----------------------------------------------------------------- pipeline
@dataclass
class ResultadoPreproceso:
    original: np.ndarray      # redimensionada, SIN corregir color (para medir el color real)
    normalizada: np.ndarray   # color/contraste unificados y suavizada (para segmentar)
    calidad: dict             # métricas y advertencias
    pasos: dict = field(default_factory=dict)  # parámetros que realmente se aplicaron


def preprocesar_imagen(img: np.ndarray, cfg: PreprocesoConfig | None = None,
                       cfg_calidad: CalidadConfig | None = None) -> ResultadoPreproceso:
    cfg = cfg or PreprocesoConfig()
    original = redimensionar_lado_max(img, cfg.lado_max)
    calidad = evaluar_calidad(original, cfg_calidad)

    x = original
    pasos: dict = {}
    if cfg.balance_blancos:
        x, ganancias = balance_blancos_mundo_gris(x, cfg.ganancia_max)
        pasos["ganancias_rgb"] = tuple(round(g, 3) for g in ganancias)
    if cfg.sigma_suavizado > 0:
        x = suavizar_gaussiano(x, cfg.sigma_suavizado)
    if cfg.estirar_contraste:
        x, rango = estirar_contraste(x, cfg.percentil_bajo, cfg.percentil_alto, cfg.expansion_max)
        pasos["rango_luminancia"] = tuple(round(v, 1) for v in rango)

    return ResultadoPreproceso(original=original, normalizada=x, calidad=calidad, pasos=pasos)

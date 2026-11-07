"""Lectura y escritura de imágenes.

Pillow se usa SOLO para decodificar/guardar archivos (JPEG, PNG...).
Todo el procesamiento posterior es NumPy: una imagen es un arreglo `uint8` de forma (alto, ancho, 3) en RGB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_TAG_ORIENTACION = 274  # etiqueta EXIF "Orientation"


def aplicar_orientacion_exif(img: np.ndarray, orientacion: int) -> np.ndarray:
    """Deja la imagen derecha según la etiqueta EXIF de orientación (valores 1 a 8).

    Los celulares guardan los píxeles tal como los captó el sensor y anotan en los metadatos
    cómo girarlos; si se ignora, una foto vertical puede llegar acostada.
    """
    if orientacion == 2:    # espejo horizontal
        return img[:, ::-1]
    if orientacion == 3:    # giro de 180°
        return img[::-1, ::-1]
    if orientacion == 4:    # espejo vertical
        return img[::-1]
    if orientacion == 5:    # transpuesta
        return img.transpose(1, 0, 2)
    if orientacion == 6:    # girar 90° en sentido horario
        return np.rot90(img, k=-1)
    if orientacion == 7:    # transpuesta + giro de 180°
        return img.transpose(1, 0, 2)[::-1, ::-1]
    if orientacion == 8:    # girar 90° en sentido antihorario
        return np.rot90(img, k=1)
    return img


def cargar_imagen(ruta: str | Path) -> np.ndarray:
    """Lee una imagen como RGB uint8, con la orientación EXIF ya corregida.

    Las imágenes con transparencia se componen sobre fondo blanco (en vez de quedar negras).
    """
    with Image.open(ruta) as im:
        try:
            orientacion = int(im.getexif().get(_TAG_ORIENTACION, 1))
        except Exception:  # metadatos dañados: se asume orientación normal
            orientacion = 1
        tiene_alfa = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        if tiene_alfa:
            rgba = im.convert("RGBA")
            fondo = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            fondo.alpha_composite(rgba)
            im = fondo.convert("RGB")
        else:
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return np.ascontiguousarray(aplicar_orientacion_exif(arr, orientacion))


def guardar_imagen(img: np.ndarray, ruta: str | Path) -> None:
    """Guarda un arreglo uint8 (RGB, gris o RGBA) creando las carpetas que falten."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(ruta)


def listar_imagenes(ruta: str | Path) -> list[Path]:
    """Un archivo -> [archivo]. Una carpeta -> todas sus imágenes, incluidas las subcarpetas, ordenadas.

    Las máscaras hechas a mano (`*_gt.png`) no son fotos y se ignoran."""
    ruta = Path(ruta)
    if ruta.is_file():
        return [ruta]
    return sorted(p for p in ruta.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES
                  and not p.stem.lower().endswith("_gt"))

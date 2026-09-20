"""
Paso 4 - Módulo de carga y redimensionamiento de imágenes.

Este módulo es la puerta de entrada del pipeline: todo lo que venga después
(filtro, Otsu, morfología, contornos) recibe la imagen desde aquí.
"""
from pathlib import Path

import cv2
import numpy as np

# Extensiones que aceptamos como imagen.
EXTENSIONES = {".jpg", ".jpeg", ".png"}

# Tamaño (en píxeles) del lado MÁS LARGO tras redimensionar.
# Es un parámetro del pipeline: cambia el resultado de la segmentación,
# así que lo dejamos como constante fácil de ajustar en las pruebas.
LADO_MAX = 500


def cargar_imagen(ruta):
    """
    Lee una imagen a color en formato BGR (el orden de OpenCV).
    Devuelve None si el archivo no existe o no se puede leer.

    Se usa np.fromfile + cv2.imdecode en lugar de cv2.imread porque imread
    falla en Windows cuando la ruta tiene tildes o caracteres especiales.
    imdecode con IMREAD_COLOR también aplica la orientación EXIF, así que las
    fotos del celular quedan derechas.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        return None
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)  # None si el formato no se entiende


def redimensionar(img, lado_max=LADO_MAX):
    """
    Reduce la imagen para que su lado más largo mida `lado_max` píxeles,
    conservando la proporción (alto/ancho). Nunca amplía.

    Devuelve (imagen_redimensionada, factor_de_escala).
    """
    alto, ancho = img.shape[:2]          # shape es (alto, ancho, canales)
    escala = lado_max / max(alto, ancho)
    if escala >= 1:                      # ya es pequeña: no ampliar
        return img.copy(), 1.0
    nuevo_tamano = (round(ancho * escala), round(alto * escala))  # cv2.resize pide (ancho, alto)
    # INTER_AREA es la interpolación recomendada para reducir imágenes.
    return cv2.resize(img, nuevo_tamano, interpolation=cv2.INTER_AREA), escala


def cargar_y_redimensionar(ruta, lado_max=LADO_MAX):
    """Atajo del pipeline: carga + redimensiona. Lanza ValueError si falla la lectura."""
    img = cargar_imagen(ruta)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    return redimensionar(img, lado_max)


def listar_imagenes(carpeta):
    """Devuelve la lista ordenada de imágenes dentro de una carpeta (y sus subcarpetas)."""
    carpeta = Path(carpeta)
    return sorted(
        p for p in carpeta.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONES
    )


def a_rgb(img_bgr):
    """OpenCV usa BGR; Matplotlib espera RGB. Úsalo solo para mostrar."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

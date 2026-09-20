"""
Paso 6 - Segmentación de la prenda con umbralización de Otsu.

Recibe la imagen en grises suavizada (Paso 5) y devuelve una máscara binaria:
    255 (blanco) = prenda
      0 (negro)  = fondo
"""
import cv2
import numpy as np


def segmentar_otsu(gris):
    """
    Separa prenda y fondo con el método de Otsu.

    Otsu prueba todos los umbrales posibles (0 a 255). Para cada uno divide los
    píxeles en dos grupos (oscuros y claros) y elige el umbral que deja los dos
    grupos lo más compactos y separados posible. Por eso funciona bien cuando el
    histograma tiene dos "montañas" con un valle en medio.

    Devuelve (umbral, mascara).
    """
    # El 0 es un valor de relleno: con THRESH_OTSU, OpenCV calcula el umbral solo.
    umbral, mascara = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu pone en blanco los píxeles MÁS CLAROS que el umbral. Pero la prenda puede
    # ser más oscura que el fondo (buzo negro) o más clara (camisa blanca sobre gris).
    # Suposición del proyecto: el fondo es lo que domina el borde de la foto.
    borde = np.concatenate([mascara[0, :], mascara[-1, :], mascara[:, 0], mascara[:, -1]])
    if borde.mean() > 127:          # el borde quedó mayormente blanco -> ese blanco es el fondo
        mascara = cv2.bitwise_not(mascara)

    return umbral, mascara

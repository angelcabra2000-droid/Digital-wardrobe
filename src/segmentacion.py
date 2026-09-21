"""
Paso 6 - Segmentación de la prenda con umbralización de Otsu.

Implementado "a mano" (sin cv2.threshold): aquí se programa el propio
algoritmo de Otsu recorriendo los 256 umbrales posibles.

Recibe la imagen en grises suavizada (Paso 5) y devuelve una máscara binaria:
    255 (blanco) = prenda
      0 (negro)  = fondo
"""
import numpy as np


def _umbral_otsu(gris):
    """
    Calcula el umbral de Otsu a partir del histograma de la imagen.

    Otsu prueba los 256 umbrales posibles (0 a 255). Para cada uno divide los
    píxeles en dos grupos -oscuros (<= umbral) y claros (> umbral)- y calcula
    qué tan separados están esos dos grupos entre sí (la "varianza entre
    clases"). El umbral elegido es el que maximiza esa separación: por eso
    funciona bien cuando el histograma tiene dos "montañas" con un valle en
    medio (típico de una prenda sobre un fondo uniforme).
    """
    # Histograma: cuántos píxeles hay de cada nivel de gris (0 a 255).
    histograma = np.bincount(gris.ravel(), minlength=256).astype(np.float64)
    total_pixeles = gris.size
    probabilidad = histograma / total_pixeles
    niveles = np.arange(256)

    media_total = np.sum(niveles * probabilidad)

    mejor_umbral = 0
    mejor_varianza = -1.0
    peso_oscuros = 0.0     # fracción de píxeles en el grupo "oscuro" (<= t)
    suma_oscuros = 0.0     # suma acumulada de nivel*probabilidad en ese grupo

    for t in range(256):
        peso_oscuros += probabilidad[t]
        peso_claros = 1.0 - peso_oscuros

        if peso_oscuros == 0.0 or peso_claros == 0.0:
            # Uno de los dos grupos quedaría vacío con este umbral: se descarta.
            suma_oscuros += t * probabilidad[t]
            continue

        suma_oscuros += t * probabilidad[t]
        media_oscuros = suma_oscuros / peso_oscuros
        media_claros = (media_total - suma_oscuros) / peso_claros

        # Varianza entre clases: qué tan separadas están las dos medias,
        # ponderado por el tamaño de cada grupo.
        varianza_entre = peso_oscuros * peso_claros * (media_oscuros - media_claros) ** 2

        if varianza_entre > mejor_varianza:
            mejor_varianza = varianza_entre
            mejor_umbral = t

    return mejor_umbral


def segmentar_otsu(gris):
    """
    Separa prenda y fondo con el método de Otsu, implementado a mano.

    Devuelve (umbral, mascara).
    """
    umbral = _umbral_otsu(gris)

    # Binarización manual: blanco (255) para los píxeles MÁS CLAROS que el
    # umbral, negro (0) para el resto. (Equivalente a cv2.THRESH_BINARY.)
    mascara = np.where(gris > umbral, 255, 0).astype(np.uint8)

    # Otsu pone en blanco los píxeles más claros que el umbral. Pero la prenda
    # puede ser más oscura que el fondo (buzo negro) o más clara (camisa
    # blanca sobre gris). Suposición del proyecto: el fondo es lo que domina
    # el borde de la foto.
    borde = np.concatenate([mascara[0, :], mascara[-1, :], mascara[:, 0], mascara[:, -1]])
    if borde.mean() > 127:          # el borde quedó mayormente blanco -> ese blanco es el fondo
        mascara = 255 - mascara     # invertir manualmente (equivalente a cv2.bitwise_not)

    return umbral, mascara
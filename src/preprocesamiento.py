"""
Paso 5 - Preprocesamiento: filtro Gaussiano y conversión a escala de grises.

Implementado "a mano" (sin cv2.GaussianBlur ni cv2.cvtColor): aquí se programa
la propia matemática del procesamiento de imágenes. Solo se usa numpy como
álgebra de arreglos (sumas, indexado, padding), nunca como atajo del algoritmo.

Recibe la imagen a color que entrega el Paso 4 y deja lista una imagen de un
solo canal (matriz 2D de 0 a 255) para que Otsu la pueda segmentar en el Paso 6.
"""
import numpy as np

from cargar_imagen import LADO_MAX, redimensionar

# Tamaño inicial del filtro Gaussiano (en píxeles). Debe ser impar.
# Se ajustará con pruebas, como indica el documento del proyecto.
KERNEL = 5


def _kernel_gaussiano_1d(tamano, sigma=None):
    """
    Genera los pesos de una Gaussiana 1D de `tamano` puntos, ya normalizados
    (suman 1). Si no se da sigma, se usa la misma fórmula que OpenCV aplica
    cuando se le pasa sigma=0: a mayor kernel, más "ancha" la campana y más
    borrosa queda la imagen.
    """
    if sigma is None or sigma <= 0:
        sigma = 0.3 * ((tamano - 1) * 0.5 - 1) + 0.8

    centro = (tamano - 1) / 2.0
    posiciones = np.arange(tamano) - centro
    pesos = np.exp(-(posiciones ** 2) / (2 * sigma ** 2))
    pesos = pesos / pesos.sum()          # normalizar para no aclarar/oscurecer la imagen
    return pesos


def _convolucion_1d(matriz, pesos, eje):
    """
    Convoluciona `matriz` (2D) con el kernel 1D `pesos`, a lo largo de un solo
    eje (0 = filas/vertical, 1 = columnas/horizontal).

    Se usa relleno por reflejo en los bordes (en vez de rellenar con ceros)
    para que la imagen no se oscurezca artificialmente en los bordes.

    La convolución se hace desplazando la imagen una posición por cada peso
    del kernel y sumando: es la misma operación matemática que hacer, para
    cada píxel, el promedio ponderado de su ventana de vecinos, pero escrita
    de forma vectorizada en vez de con un bucle por píxel.
    """
    radio = len(pesos) // 2
    if eje == 1:  # horizontal
        relleno = np.pad(matriz, ((0, 0), (radio, radio)), mode="reflect").astype(np.float64)
        salida = np.zeros(matriz.shape, dtype=np.float64)
        ancho = matriz.shape[1]
        for i, peso in enumerate(pesos):
            salida += peso * relleno[:, i:i + ancho]
    else:  # vertical
        relleno = np.pad(matriz, ((radio, radio), (0, 0)), mode="reflect").astype(np.float64)
        salida = np.zeros(matriz.shape, dtype=np.float64)
        alto = matriz.shape[0]
        for i, peso in enumerate(pesos):
            salida += peso * relleno[i:i + alto, :]
    return salida


def suavizar(img, kernel=KERNEL):
    """
    Filtro Gaussiano hecho a mano: reemplaza cada píxel por un promedio
    ponderado de sus vecinos (más peso al centro, menos a los bordes de la
    ventana kernel x kernel). Sirve para reducir ruido leve y la textura fina
    de la tela, antes de convertir a grises y binarizar.

    La Gaussiana 2D es separable: convolucionar primero en horizontal y luego
    en vertical con el mismo kernel 1D da el mismo resultado que convolucionar
    con el kernel 2D completo, pero es mucho más rápido.
    """
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("El kernel debe ser un número impar (3, 5, 7, ...)")

    pesos = _kernel_gaussiano_1d(kernel)

    if img.ndim == 3:  # imagen a color: suavizar cada canal por separado
        canales = []
        for c in range(img.shape[2]):
            horizontal = _convolucion_1d(img[:, :, c], pesos, eje=1)
            vertical = _convolucion_1d(horizontal, pesos, eje=0)
            canales.append(vertical)
        suave = np.stack(canales, axis=-1)
    else:  # imagen en grises: un solo canal
        horizontal = _convolucion_1d(img, pesos, eje=1)
        suave = _convolucion_1d(horizontal, pesos, eje=0)

    return np.clip(np.round(suave), 0, 255).astype(np.uint8)


def a_grises(img_bgr):
    """
    Convierte de color (3 canales, orden BGR) a grises (1 canal), a mano:
    se combinan los tres canales con los pesos estándar de luminancia
        0.299*R + 0.587*G + 0.114*B
    (el ojo humano percibe más el verde que el azul o el rojo).
    """
    azul = img_bgr[:, :, 0].astype(np.float64)
    verde = img_bgr[:, :, 1].astype(np.float64)
    rojo = img_bgr[:, :, 2].astype(np.float64)

    gris = 0.299 * rojo + 0.587 * verde + 0.114 * azul
    return np.round(gris).astype(np.uint8)


def preprocesar(img_bgr, lado_max=LADO_MAX, kernel=KERNEL):
    """
    Pipeline completo hasta este paso: redimensionar -> Gaussiano -> grises.
    Devuelve (imagen_color_redimensionada, imagen_gris_suavizada).
    """
    color, _ = redimensionar(img_bgr, lado_max)
    gris = a_grises(suavizar(color, kernel))
    return color, gris
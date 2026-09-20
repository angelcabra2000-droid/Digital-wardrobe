"""
Paso 5 - Preprocesamiento: filtro Gaussiano y conversión a escala de grises.

Recibe la imagen a color que entrega el Paso 4 y deja lista una imagen de un
solo canal (matriz 2D de 0 a 255) para que Otsu la pueda segmentar en el Paso 6.
"""
import cv2

from cargar_imagen import LADO_MAX, redimensionar

# Tamaño inicial del filtro Gaussiano (en píxeles). Debe ser impar.
# Se ajustará con pruebas, como indica el documento del proyecto.
KERNEL = 5


def suavizar(img, kernel=KERNEL):
    """
    Filtro Gaussiano: reemplaza cada píxel por un promedio ponderado de sus vecinos
    (más peso al centro, menos a los bordes de la ventana kernel x kernel).
    Sirve para reducir ruido leve y la textura fina de la tela.

    Con sigma=0, OpenCV calcula automáticamente la "fuerza" del suavizado a partir
    del tamaño del kernel: a mayor kernel, más borroso.
    """
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("El kernel debe ser un número impar (3, 5, 7, ...)")
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def a_grises(img_bgr):
    """
    Convierte de color (3 canales) a grises (1 canal).
    OpenCV combina los canales con pesos: 0.299*R + 0.587*G + 0.114*B
    (el ojo humano percibe más el verde que el azul).
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def preprocesar(img_bgr, lado_max=LADO_MAX, kernel=KERNEL):
    """
    Pipeline completo hasta este paso: redimensionar -> Gaussiano -> grises.
    Devuelve (imagen_color_redimensionada, imagen_gris_suavizada).
    """
    color, _ = redimensionar(img_bgr, lado_max)
    gris = a_grises(suavizar(color, kernel))
    return color, gris

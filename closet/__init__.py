"""Closet Digital: segmentación y clasificación de prendas con procesamiento de imágenes puro.

Pipeline (cada paso es un módulo):
    0. dataset / revision                     : inventariar y dividir la base       <- paso 0
    1. imagen_io / preprocesamiento / calidad : leer, unificar y revisar la foto   <- paso 1
    2. segmentacion                           : K-means + regiones conectadas       (paso 2)
    3. normalizacion                          : recortar, alinear y escalar         (paso 3)
    4. caracteristicas                        : color, forma, textura (Haralick)    (paso 4)
    5. clasificacion                          : tipo de prenda y atributos          (paso 5)
"""

__version__ = "0.2.0"

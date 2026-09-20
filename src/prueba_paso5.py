"""
Prueba del Paso 5.

Uso (desde la raíz del proyecto, con el entorno virtual activado):
    python src\\prueba_paso5.py                   -> usa la primera imagen de data\\raw
    python src\\prueba_paso5.py ruta\\foto.jpg    -> usa esa imagen
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cargar_imagen import a_rgb, cargar_imagen, listar_imagenes, redimensionar
from preprocesamiento import a_grises, preprocesar, suavizar

RAIZ = Path(__file__).resolve().parent.parent

if len(sys.argv) > 1:
    ruta = Path(sys.argv[1])
else:
    imagenes = listar_imagenes(RAIZ / "data" / "raw")
    if not imagenes:
        sys.exit("No hay imágenes en data/raw")
    ruta = imagenes[0]

original = cargar_imagen(ruta)
if original is None:
    sys.exit(f"No se pudo leer {ruta}")

color, gris_suave = preprocesar(original)          # kernel 5x5 por defecto
gris_sin_filtro = a_grises(color)                   # para comparar
gris_k15 = a_grises(suavizar(color, 15))            # kernel exagerado, solo para ver el efecto

print(f"Imagen: {ruta.name}")
print(f"Color : {color.shape}   (alto, ancho, canales)")
print(f"Grises: {gris_suave.shape}   (alto, ancho)  tipo={gris_suave.dtype}  "
      f"min={gris_suave.min()}  max={gris_suave.max()}")

# --- Ventana 1: imágenes ---


def zona_con_mas_detalle(gris, mitad=40):
    """Busca la zona con más bordes/textura (para que el zoom muestre algo útil)."""
    gx = cv2.Sobel(gris, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gris, cv2.CV_32F, 0, 1)
    actividad = cv2.blur(cv2.magnitude(gx, gy), (41, 41))
    cy, cx = np.unravel_index(actividad.argmax(), actividad.shape)
    alto, ancho = gris.shape
    cy = min(max(cy, mitad), alto - mitad)
    cx = min(max(cx, mitad), ancho - mitad)
    return cy - mitad, cy + mitad, cx - mitad, cx + mitad


y0, y1, x0, x1 = zona_con_mas_detalle(gris_sin_filtro)

fig1, ejes = plt.subplots(2, 3, figsize=(12, 8))
ejes[0, 0].imshow(a_rgb(color));                          ejes[0, 0].set_title("Color (Paso 4)")
ejes[0, 1].imshow(gris_sin_filtro, cmap="gray");          ejes[0, 1].set_title("Grises, sin filtro")
ejes[0, 2].imshow(gris_suave, cmap="gray");               ejes[0, 2].set_title("Grises + Gaussiano 5x5")
ejes[1, 0].imshow(gris_sin_filtro[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
ejes[1, 0].set_title("Zoom (zona con más detalle): sin filtro")
ejes[1, 1].imshow(gris_suave[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
ejes[1, 1].set_title("Zoom: kernel 5x5")
ejes[1, 2].imshow(gris_k15[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
ejes[1, 2].set_title("Zoom: kernel 15x15 (exagerado)")
for e in ejes.ravel():
    e.axis("off")
fig1.tight_layout()

# --- Ventana 2: histograma (lo que usará Otsu en el Paso 6) ---
fig2, eje_h = plt.subplots(figsize=(7, 4))
eje_h.hist(gris_suave.ravel(), bins=256, range=(0, 255), color="gray")
eje_h.set_title("Histograma de grises (Gaussiano 5x5)")
eje_h.set_xlabel("Nivel de gris (0 = negro, 255 = blanco)")
eje_h.set_ylabel("Cantidad de píxeles")
fig2.tight_layout()

plt.show()

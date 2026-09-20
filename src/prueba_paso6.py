"""
Prueba del Paso 6.

Uso (desde la raíz del proyecto, con el entorno virtual activado).
Si el nombre del archivo tiene espacios o paréntesis, ponlo entre comillas:
    python src\\prueba_paso6.py                                       -> primera imagen de data\\raw
    python src\\prueba_paso6.py "data\\raw\\camisas\\camisa_ (10).jpg"
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cargar_imagen import a_rgb, cargar_imagen, listar_imagenes
from preprocesamiento import preprocesar
from segmentacion import segmentar_otsu

RAIZ = Path(__file__).resolve().parent.parent

if len(sys.argv) > 1:
    ruta = Path(sys.argv[1])
else:
    imagenes = listar_imagenes(RAIZ / "data" / "raw")
    if not imagenes:
        sys.exit("No hay imágenes en data/raw")
    ruta = imagenes[0]

if not ruta.is_file():
    sys.exit(f"No existe el archivo: {ruta}\n"
             "Si el nombre tiene espacios o paréntesis, escribe la ruta entre comillas.")

original = cargar_imagen(ruta)
if original is None:
    sys.exit(f"No se pudo leer {ruta} (¿está dañada o no es una imagen?)")

color, gris = preprocesar(original)
umbral, mascara = segmentar_otsu(gris)

porcentaje = (mascara > 0).mean() * 100
print(f"Imagen : {ruta.name}")
print(f"Umbral de Otsu : {umbral:.0f}")
print(f"La máscara marca {porcentaje:.1f} % de la imagen como prenda")

# Prenda recortada sobre gris medio (para ver qué se quedó la máscara)
recorte = np.full_like(color, 128)
recorte[mascara > 0] = color[mascara > 0]

fig, ejes = plt.subplots(2, 2, figsize=(11, 9))
ejes[0, 0].imshow(a_rgb(color))
ejes[0, 0].set_title("Color (Paso 4)")
ejes[0, 0].axis("off")

ejes[0, 1].hist(gris.ravel(), bins=256, range=(0, 255), color="gray")
ejes[0, 1].axvline(umbral, color="red", linewidth=2, label=f"Umbral Otsu = {umbral:.0f}")
ejes[0, 1].set_title("Histograma y umbral elegido")
ejes[0, 1].set_xlabel("Nivel de gris")
ejes[0, 1].legend()

ejes[1, 0].imshow(mascara, cmap="gray")
ejes[1, 0].set_title("Máscara (blanco = prenda)")
ejes[1, 0].axis("off")

ejes[1, 1].imshow(a_rgb(recorte))
ejes[1, 1].set_title("Prenda recortada con la máscara")
ejes[1, 1].axis("off")

fig.tight_layout()
plt.show()

"""
Prueba del Paso 4.

Uso (desde la raíz del proyecto, con el entorno virtual activado):
    python src\\prueba_paso4.py                  -> revisa todo data\\raw y muestra la primera imagen
    python src\\prueba_paso4.py ruta\\foto.jpg   -> muestra esa imagen en tres tamaños
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from cargar_imagen import a_rgb, cargar_imagen, listar_imagenes, redimensionar

RAIZ = Path(__file__).resolve().parent.parent   # carpeta del proyecto
LADOS_A_COMPARAR = [800, 500, 300]

# 1) Revisión de todo el dataset
imagenes = listar_imagenes(RAIZ / "data" / "raw")
print(f"{len(imagenes)} imágenes encontradas en data/raw\n")

for ruta in imagenes:
    img = cargar_imagen(ruta)
    if img is None:
        print(f"NO SE PUDO LEER: {ruta.name}")
        continue
    peq, escala = redimensionar(img)
    print(f"{ruta.parent.name}/{ruta.name}: "
          f"{img.shape[1]}x{img.shape[0]} -> {peq.shape[1]}x{peq.shape[0]} (escala {escala:.3f})")

# 2) Comparación visual de tamaños sobre UNA imagen
if len(sys.argv) > 1:
    elegida = Path(sys.argv[1])
elif imagenes:
    elegida = imagenes[0]
else:
    sys.exit("No hay imágenes para mostrar.")

original = cargar_imagen(elegida)
if original is None:
    sys.exit(f"No se pudo leer {elegida}")

fig, ejes = plt.subplots(2, len(LADOS_A_COMPARAR), figsize=(12, 9))
alto_orig, ancho_orig = original.shape[:2]
for col, lado in enumerate(LADOS_A_COMPARAR):
    peq, _ = redimensionar(original, lado)
    alto, ancho = peq.shape[:2]

    # Fila 1: imagen completa. Los ejes muestran píxeles reales.
    titulo = f"lado mayor = {lado} px\n{ancho}x{alto}"
    if max(alto, ancho) < lado:
        titulo += "\n(la original ya es menor: no se amplía)"
    ejes[0, col].imshow(a_rgb(peq))
    ejes[0, col].set_title(titulo)

    # Fila 2: zoom a la MISMA zona relativa, píxel por píxel (sin suavizar).
    y0, y1 = int(alto * 0.35), int(alto * 0.55)
    x0, x1 = int(ancho * 0.30), int(ancho * 0.60)
    ejes[1, col].imshow(a_rgb(peq[y0:y1, x0:x1]), interpolation="nearest")
    ejes[1, col].set_title("zoom a la misma zona")

plt.tight_layout()
plt.show()
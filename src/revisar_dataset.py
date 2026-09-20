import cv2
from pathlib import Path

RAIZ = Path("data")

for carpeta in ["raw", "test"]:
    for categoria in ["chaquetas", "camisas", "pantalones"]:
        ruta = RAIZ / carpeta / categoria
        archivos = sorted(ruta.glob("*.*"))
        print(f"\n{carpeta}/{categoria}: {len(archivos)} archivos")
        for archivo in archivos:
            img = cv2.imread(str(archivo))
            if img is None:
                print(f"  NO SE PUDO LEER: {archivo.name}")
            else:
                alto, ancho = img.shape[:2]
                print(f"  {archivo.name}: {ancho}x{alto}")
"""
revisar_segmentacion.py
-------------------------
Herramienta de CALIBRACIÓN, no de pruebas automatizadas. A diferencia
de test_segmentation.py (que solo verifica que nada explote y que los
números caigan en rangos razonables), este script genera algo que
puedas MIRAR: para cada foto en data/, guarda una imagen con la
máscara ganadora superpuesta en rojo semi-transparente, y un reporte
de texto con el desglose de puntajes de las 4 candidatas.

Por qué hace falta esto además de test_segmentation.py:
Una máscara puede tener "área razonable" (10-100%) y aun así estar
segmentando la mitad del fondo en vez de la prenda completa, o dejar
un hueco en media manga por un reflejo de tela. Esos errores solo se
detectan a simple vista, no con un assert.

Cómo correrlo (desde src/):
    python revisar_segmentacion.py

Genera:
    data/_debug_segmentacion/<nombre>_overlay.jpg   (una por foto)
    data/_debug_segmentacion/reporte.txt             (puntajes de todas)
"""

import glob
import os

import cv2
import numpy as np

from preprocessing import prepare_for_display, preprocess_image
from segmentation import segment_garment

CARPETA_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CARPETA_SALIDA = os.path.join(CARPETA_DATA, "_debug_segmentacion")

COLOR_OVERLAY = np.array([255, 0, 0], dtype=np.float64)  # rojo, en RGB
OPACIDAD = 0.45


def _dibujar_overlay(imagen_display: np.ndarray, mask_prenda: np.ndarray) -> np.ndarray:
    """
    Mezcla la imagen NÍTIDA (prepare_for_display, sin ecualizar ni
    desenfocar) con un tinte rojo semi-transparente donde mask_prenda
    es True, para poder juzgar a simple vista si la segmentación
    coincide con la prenda real.

    Esto es puramente para depurar visualmente -- no forma parte del
    algoritmo de segmentación ni se usa en el pipeline real.
    """
    overlay = imagen_display.astype(np.float64).copy()
    overlay[mask_prenda] = (
        overlay[mask_prenda] * (1 - OPACIDAD) + COLOR_OVERLAY * OPACIDAD
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    extensiones = ("*.jpg", "*.jpeg", "*.png")
    fotos = []
    for ext in extensiones:
        patron = os.path.join(CARPETA_DATA, "**", ext)
        fotos.extend(
            p for p in glob.glob(patron, recursive=True)
            if CARPETA_SALIDA not in p  # no reprocesar overlays de corridas anteriores
        )

    if not fotos:
        print(f"No se encontraron fotos en {os.path.abspath(CARPETA_DATA)}")
        return

    lineas_reporte = []

    for path in sorted(fotos):
        nombre = os.path.basename(path)
        try:
            imagen, mask_valida = preprocess_image(path)
            imagen_display, _ = prepare_for_display(path)
            mask_prenda, candidatas = segment_garment(imagen, mask_valida, debug=True)

            overlay = _dibujar_overlay(imagen_display, mask_prenda)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            ruta_overlay = os.path.join(CARPETA_SALIDA, f"{os.path.splitext(nombre)[0]}_overlay.jpg")
            cv2.imwrite(ruta_overlay, overlay_bgr)

            porcentaje = 100 * mask_prenda.sum() / mask_valida.sum() if mask_valida.sum() else 0
            ganadora = max(candidatas, key=lambda c: c["puntaje_total"])

            lineas_reporte.append(f"\n=== {nombre} ===")
            lineas_reporte.append(f"  Ganadora: {ganadora['nombre']}  (área final tras BFS: {porcentaje:.1f}%)")
            for c in sorted(candidatas, key=lambda c: -c["puntaje_total"]):
                marca = " <-- GANADORA" if c is ganadora else ""
                lineas_reporte.append(
                    f"    {c['nombre']:<20} umbral={c['umbral']:>3}  "
                    f"area={c['area_proporcion']:.2%}  "
                    f"compacidad={c['compacidad']:.3f}  "
                    f"borde={c['ratio_contacto_borde']:.2%}  "
                    f"puntaje_total={c['puntaje_total']:.3f}{marca}"
                )
            print(f"[OK] {nombre} -> {ganadora['nombre']} ({porcentaje:.1f}%)")

        except Exception as e:
            lineas_reporte.append(f"\n=== {nombre} ===\n  ERROR: {e}")
            print(f"[ERROR] {nombre}: {e}")

    ruta_reporte = os.path.join(CARPETA_SALIDA, "reporte.txt")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_reporte))

    print(f"\nOverlays guardados en: {os.path.abspath(CARPETA_SALIDA)}")
    print(f"Reporte de puntajes en: {os.path.abspath(ruta_reporte)}")


if __name__ == "__main__":
    main()

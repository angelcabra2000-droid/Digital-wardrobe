"""
test_preprocessing.py
----------------------
Pruebas para verificar que resize_image() y la propagación de la
máscara funcionan correctamente, además de correr el pipeline completo
sobre las fotos reales del proyecto para detectar errores.

Cómo correrlo:
    python test_preprocessing.py

No usa pytest a propósito, para mantenerlo simple: cada función de
prueba imprime PASS/FAIL y termina con un resumen. Si algo falla, el
mensaje te dice exactamente qué esperaba vs. qué obtuvo.
"""

import glob
import os
import sys

import numpy as np

from preprocessing import preprocess_image, prepare_for_display, resize_image

# Carpeta de fotos reales del proyecto: data/, un nivel arriba de src/
# (según la estructura del repo: Digital wardrobe/data/, Digital wardrobe/src/).
CARPETA_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

resultados = []


def check(nombre: str, condicion: bool, detalle: str = ""):
    estado = "PASS" if condicion else "FAIL"
    resultados.append(condicion)
    print(f"[{estado}] {nombre}" + (f" -- {detalle}" if detalle and not condicion else ""))


def test_imagen_cuadrada_no_genera_padding():
    """Si la imagen de entrada ya tiene la proporción exacta del target,
    no debería haber ningún píxel de relleno: mask debe ser 100% True."""
    imagen_cuadrada = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    canvas, mask = resize_image(imagen_cuadrada, target_size=(512, 512))

    check(
        "Imagen cuadrada -> mask 100% True",
        mask.all(),
        f"esperaba mask.all()==True, {mask.sum()}/{mask.size} True",
    )
    check(
        "Imagen cuadrada -> canvas shape correcto",
        canvas.shape == (512, 512, 3),
    )


def test_imagen_alargada_genera_padding_en_los_bordes():
    """Con una imagen muy vertical, el padding debe caer en los bordes
    izquierdo/derecho (letterbox horizontal), y el centro debe quedar
    completamente válido (True)."""
    imagen_vertical = np.random.randint(0, 255, (800, 200, 3), dtype=np.uint8)
    canvas, mask = resize_image(imagen_vertical, target_size=(512, 512))

    centro_x = mask.shape[1] // 2
    check(
        "Imagen vertical -> el centro de la máscara es válido",
        mask[:, centro_x].all(),
    )
    check(
        "Imagen vertical -> las esquinas son padding (False)",
        not mask[0, 0] and not mask[0, -1] and not mask[-1, 0] and not mask[-1, -1],
    )
    check(
        "Imagen vertical -> hay padding real (no toda la máscara es True)",
        not mask.all(),
        f"{mask.sum()}/{mask.size} True -- si esto falla, revisa el cálculo de offsets",
    )


def test_conteo_de_pixeles_validos_coincide_con_el_resize():
    """El número exacto de píxeles True debe ser new_w * new_h, el
    tamaño de la imagen ya escalada (antes de pegarla en el canvas).
    Esto detecta errores de offset de un solo píxel."""
    h, w = 600, 400
    target_w, target_h = 512, 512
    imagen = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w_esperado, new_h_esperado = int(w * scale), int(h * scale)
    pixeles_esperados = new_w_esperado * new_h_esperado

    _, mask = resize_image(imagen, target_size=(target_w, target_h))

    check(
        "Conteo de píxeles válidos == new_w * new_h",
        mask.sum() == pixeles_esperados,
        f"esperaba {pixeles_esperados}, obtuve {mask.sum()}",
    )


def test_mask_y_canvas_tienen_la_misma_forma_espacial():
    imagen = np.random.randint(0, 255, (250, 500, 3), dtype=np.uint8)
    canvas, mask = resize_image(imagen, target_size=(512, 512))
    check(
        "mask.shape coincide con canvas.shape[:2]",
        mask.shape == canvas.shape[:2],
        f"canvas={canvas.shape}, mask={mask.shape}",
    )


def test_pipeline_completo_sobre_fotos_reales():
    """No valida contenido (eso lo hará segmentation.py), solo que el
    pipeline completo corre sin errores sobre las fotos reales del
    proyecto y que el % de píxeles válidos es razonable.

    Busca de forma RECURSIVA dentro de data/ (incluyendo subcarpetas,
    como data/Fotos_Procesamiento_Imagenes/), no solo en el nivel raíz,
    para que no importe cómo organicen las fotos de prueba ahí adentro.
    """
    extensiones = ("*.jpg", "*.jpeg", "*.png")
    fotos = []
    for ext in extensiones:
        patron = os.path.join(CARPETA_DATA, "**", ext)
        fotos.extend(glob.glob(patron, recursive=True))

    if not fotos:
        print(f"[SKIP] No se encontraron fotos en {os.path.abspath(CARPETA_DATA)} ni en sus subcarpetas")
        print("       Revisa que tus fotos de prueba estén dentro de data/.")
        return

    for path in fotos:
        nombre = os.path.basename(path)
        try:
            imagen, mask = preprocess_image(path)
            _, mask_display = prepare_for_display(path)
            porcentaje_valido = 100 * mask.sum() / mask.size
            check(
                f"preprocess_image corre sin error: {nombre}",
                imagen.shape == (512, 512, 3),
            )
            check(
                f"% de píxeles válidos es razonable (10-100%): {nombre}",
                10 <= porcentaje_valido <= 100,
                f"obtuve {porcentaje_valido:.1f}%",
            )
            check(
                f"mask de preprocess_image coincide con la de prepare_for_display: {nombre}",
                np.array_equal(mask, mask_display),
            )
        except Exception as e:
            check(f"preprocess_image corre sin error: {nombre}", False, str(e))


if __name__ == "__main__":
    test_imagen_cuadrada_no_genera_padding()
    test_imagen_alargada_genera_padding_en_los_bordes()
    test_conteo_de_pixeles_validos_coincide_con_el_resize()
    test_mask_y_canvas_tienen_la_misma_forma_espacial()
    test_pipeline_completo_sobre_fotos_reales()

    total = len(resultados)
    ok = sum(resultados)
    print(f"\n{ok}/{total} pruebas pasaron")
    sys.exit(0 if ok == total else 1)
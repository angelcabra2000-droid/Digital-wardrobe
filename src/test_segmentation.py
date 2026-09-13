"""
test_segmentation.py
----------------------
Pruebas para segment_garment() y sus piezas internas (Otsu manual,
compacidad, contacto con el borde, componente conexo más grande),
además de correr el pipeline completo (preprocessing + segmentation)
sobre las fotos reales del proyecto.

Cómo correrlo:
    python test_segmentation.py

Mismo estilo que test_preprocessing.py: sin pytest, cada función de
prueba imprime PASS/FAIL y al final se muestra un resumen.
"""

import glob
import os
import sys

import numpy as np

from preprocessing import preprocess_image
from segmentation import (
    _componente_conexo_mas_grande,
    _compacidad,
    _otsu_threshold,
    _ratio_contacto_borde,
    _borde_de_region_valida,
    segment_garment,
)

CARPETA_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

resultados = []


def check(nombre: str, condicion: bool, detalle: str = ""):
    estado = "PASS" if condicion else "FAIL"
    resultados.append(condicion)
    print(f"[{estado}] {nombre}" + (f" -- {detalle}" if detalle and not condicion else ""))


def test_otsu_separa_dos_bloques_obvios():
    """Imagen sintética con un bloque claro (200) y uno oscuro (20).

    Nota: Otsu puede devolver legítimamente un umbral igual a 20 (el
    borde exacto de la meseta oscura) si no hay píxeles intermedios
    entre ambos bloques -- la varianza entre-clases queda igual de
    máxima en todo ese tramo plano del histograma, y el algoritmo se
    queda con el primer t que alcanza ese máximo. Por eso la prueba
    real no es "el umbral cae estrictamente entre 20 y 200", sino que
    el umbral efectivamente separa ambos bloques sin mezclarlos."""
    canal = np.zeros((100, 100), dtype=np.uint8)
    canal[:, :50] = 20
    canal[:, 50:] = 200
    valid = np.ones((100, 100), dtype=bool)

    umbral = _otsu_threshold(canal, valid)
    separa_correctamente = (canal[:, :50] <= umbral).all() and (canal[:, 50:] > umbral).all()

    check(
        "Otsu separa el bloque oscuro del bloque claro sin mezclarlos",
        separa_correctamente,
        f"umbral obtenido: {umbral}",
    )


def test_otsu_ignora_pixeles_invalidos():
    """Si se agrega un bloque de relleno en 0 (letterbox) pero se
    marca como inválido, el umbral NO debe verse arrastrado hacia el
    negro -- debe comportarse como si el relleno no existiera."""
    canal = np.zeros((100, 100), dtype=np.uint8)
    canal[:, :50] = 20
    canal[:, 50:] = 200

    canal_con_relleno = canal.copy()
    canal_con_relleno[:, :30] = 0  # simula letterbox negro
    valid = np.ones((100, 100), dtype=bool)
    valid[:, :30] = False  # el relleno se marca como inválido

    umbral_sin_relleno = _otsu_threshold(canal, np.ones((100, 100), dtype=bool))
    umbral_ignorando_relleno = _otsu_threshold(canal_con_relleno, valid)

    check(
        "Otsu con máscara de válidos da el mismo umbral que sin relleno",
        umbral_sin_relleno == umbral_ignorando_relleno,
        f"sin relleno={umbral_sin_relleno}, ignorando relleno={umbral_ignorando_relleno}",
    )


def test_compacidad_circulo_mayor_que_forma_dispersa():
    """Un círculo debe tener compacidad notablemente más alta que una
    máscara con la misma área pero hecha de puntos dispersos."""
    tam = 101
    yy, xx = np.mgrid[0:tam, 0:tam]
    centro = tam // 2
    circulo = (xx - centro) ** 2 + (yy - centro) ** 2 <= 30 ** 2

    dispersa = np.zeros((tam, tam), dtype=bool)
    indices_planos = np.random.choice(tam * tam, size=int(circulo.sum()), replace=False)
    dispersa.flat[indices_planos] = True

    compacidad_circulo = _compacidad(circulo)
    compacidad_dispersa = _compacidad(dispersa)

    check(
        "Círculo es más compacto que puntos dispersos",
        compacidad_circulo > compacidad_dispersa,
        f"círculo={compacidad_circulo:.3f}, dispersa={compacidad_dispersa:.3f}",
    )
    check(
        "Compacidad del círculo está en rango razonable (>0.7)",
        compacidad_circulo > 0.7,
        f"obtuve {compacidad_circulo:.3f}",
    )


def test_compacidad_mascara_vacia_es_cero():
    vacia = np.zeros((50, 50), dtype=bool)
    check("Compacidad de máscara vacía es 0.0", _compacidad(vacia) == 0.0)


def test_ratio_borde_prenda_centrada_es_bajo():
    """Una 'prenda' cuadrada centrada, lejos del borde, debe tener
    ratio de contacto con el borde igual a 0."""
    valid = np.ones((100, 100), dtype=bool)
    borde = _borde_de_region_valida(valid)

    prenda_centrada = np.zeros((100, 100), dtype=bool)
    prenda_centrada[30:70, 30:70] = True

    ratio = _ratio_contacto_borde(prenda_centrada, borde)
    check(
        "Prenda centrada no toca el borde real de la foto",
        ratio == 0.0,
        f"obtuve {ratio:.3f}",
    )


def test_ratio_borde_fondo_que_toca_bordes_es_alto():
    """Simula el caso de que el umbral quedó invertido: la candidata
    'prenda' en realidad es el fondo, que sí toca los bordes."""
    valid = np.ones((100, 100), dtype=bool)
    borde = _borde_de_region_valida(valid)

    fondo_como_candidata = np.ones((100, 100), dtype=bool)
    fondo_como_candidata[30:70, 30:70] = False  # el "hueco" sería la prenda real

    ratio = _ratio_contacto_borde(fondo_como_candidata, borde)
    check(
        "Candidata que en realidad es fondo tiene ratio de borde alto",
        ratio > 0.9,
        f"obtuve {ratio:.3f}",
    )


def test_borde_de_region_valida_respeta_letterbox():
    """Con letterbox horizontal (padding a los lados), el borde real
    de la foto debe caer en la frontera válido/inválido, NO en los
    bordes izquierdo/derecho del canvas completo."""
    valid = np.zeros((100, 100), dtype=bool)
    valid[:, 20:80] = True  # región válida centrada, con padding a los lados

    borde = _borde_de_region_valida(valid)

    check(
        "El borde real no incluye el borde izquierdo/derecho del canvas (eso es padding)",
        not borde[:, 0].any() and not borde[:, 99].any(),
    )
    check(
        "El borde real sí marca la frontera entre región válida y padding",
        borde[:, 20].any() and borde[:, 79].any(),
    )


def test_componente_conexo_se_queda_con_el_mas_grande():
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:15, 5:15] = True  # componente grande: 100 píxeles
    mask[40:43, 40:43] = True  # componente pequeño (ruido): 9 píxeles

    limpia = _componente_conexo_mas_grande(mask)

    check(
        "Se conserva el componente grande completo",
        limpia[5:15, 5:15].all(),
    )
    check(
        "Se elimina el componente pequeño (ruido)",
        not limpia[40:43, 40:43].any(),
    )
    check(
        "El tamaño final coincide con el del componente grande",
        limpia.sum() == 100,
        f"esperaba 100, obtuve {limpia.sum()}",
    )


def test_segment_garment_ignora_letterbox_sintetico():
    """Imagen sintética: fondo gris claro uniforme con un cuadrado
    oscuro centrado (la 'prenda'), rodeada de letterbox negro marcado
    como inválido. La prenda segmentada NO debe incluir el relleno,
    y su área debe aproximarse al cuadrado real."""
    canvas = np.full((200, 200, 3), 180, dtype=np.uint8)  # fondo claro
    canvas[60:140, 60:140] = 30  # "prenda" oscura centrada

    valid = np.ones((200, 200), dtype=bool)
    valid[:, :20] = False  # letterbox simulado a la izquierda
    valid[:, -20:] = False  # y a la derecha
    canvas[:, :20] = 0
    canvas[:, -20:] = 0

    mask_prenda = segment_garment(canvas, valid)

    check(
        "La máscara final no incluye ningún píxel de letterbox",
        not mask_prenda[:, :20].any() and not mask_prenda[:, -20:].any(),
    )

    area_esperada = 80 * 80
    area_obtenida = int(mask_prenda.sum())
    check(
        "El área segmentada se aproxima al cuadrado real (±15%)",
        abs(area_obtenida - area_esperada) / area_esperada < 0.15,
        f"esperaba ~{area_esperada}, obtuve {area_obtenida}",
    )


def test_pipeline_completo_sobre_fotos_reales():
    """No valida el contenido exacto de la máscara (eso requiere ojo
    humano o ground truth), solo que preprocessing + segmentation
    corren sin errores sobre las fotos reales y que el resultado tiene
    forma/tamaño razonable."""
    extensiones = ("*.jpg", "*.jpeg", "*.png")
    fotos = []
    for ext in extensiones:
        patron = os.path.join(CARPETA_DATA, "**", ext)
        fotos.extend(glob.glob(patron, recursive=True))

    if not fotos:
        print(f"[SKIP] No se encontraron fotos en {os.path.abspath(CARPETA_DATA)} ni en sus subcarpetas")
        return

    for path in fotos:
        nombre = os.path.basename(path)
        try:
            imagen, mask_valida = preprocess_image(path)
            mask_prenda = segment_garment(imagen, mask_valida)

            check(
                f"segment_garment corre sin error: {nombre}",
                mask_prenda.shape == mask_valida.shape,
            )
            check(
                f"máscara final no incluye píxeles de letterbox: {nombre}",
                not (mask_prenda & ~mask_valida).any(),
            )
            porcentaje = 100 * mask_prenda.sum() / mask_valida.sum() if mask_valida.sum() else 0
            check(
                f"área segmentada es razonable (1-95% de la región válida): {nombre}",
                1 <= porcentaje <= 95,
                f"obtuve {porcentaje:.1f}%",
            )
        except Exception as e:
            check(f"segment_garment corre sin error: {nombre}", False, str(e))


if __name__ == "__main__":
    test_otsu_separa_dos_bloques_obvios()
    test_otsu_ignora_pixeles_invalidos()
    test_compacidad_circulo_mayor_que_forma_dispersa()
    test_compacidad_mascara_vacia_es_cero()
    test_ratio_borde_prenda_centrada_es_bajo()
    test_ratio_borde_fondo_que_toca_bordes_es_alto()
    test_borde_de_region_valida_respeta_letterbox()
    test_componente_conexo_se_queda_con_el_mas_grande()
    test_segment_garment_ignora_letterbox_sintetico()
    test_pipeline_completo_sobre_fotos_reales()

    total = len(resultados)
    ok = sum(resultados)
    print(f"\n{ok}/{total} pruebas pasaron")
    sys.exit(0 if ok == total else 1)

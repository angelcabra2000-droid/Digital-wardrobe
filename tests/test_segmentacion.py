import numpy as np
import pytest

from closet.config import SegmentacionConfig
from closet.segmentacion import (cerrar_caja, componentes_por_etiqueta, filtro_caja, kmeans, rellenar_huecos,
                                 rgb_a_lab, segmentar_imagen, recortar_fondo)


def foto_sintetica(alto=200, ancho=160, color_fondo=(235, 235, 235), color_prenda=(180, 40, 40)):
    img = np.empty((alto, ancho, 3), dtype=np.uint8)
    img[:] = color_fondo
    img[40:170, 40:120] = color_prenda
    gt = np.zeros((alto, ancho), dtype=bool)
    gt[40:170, 40:120] = True
    return img, gt


def test_lab_de_blanco_y_negro():
    lab = rgb_a_lab(np.array([[255.0, 255.0, 255.0], [0.0, 0.0, 0.0]]))
    assert lab[0, 0] == pytest.approx(100, abs=0.1) and abs(lab[0, 1]) < 0.1 and abs(lab[0, 2]) < 0.1
    assert lab[1, 0] == pytest.approx(0, abs=0.1)


def test_filtro_caja_conserva_constantes():
    assert np.allclose(filtro_caja(np.full((20, 30), 7.0), 3), 7.0)


def test_kmeans_separa_dos_grupos_y_es_reproducible():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.2, (100, 2)), rng.normal(10, 0.2, (100, 2))])
    e1, c1, _ = kmeans(X, 2, seed=3)
    e2, _, _ = kmeans(X, 2, seed=3)
    assert np.array_equal(e1, e2)
    assert len(set(e1[:100])) == 1 and len(set(e1[100:])) == 1 and e1[0] != e1[-1]
    assert sorted(np.round(c1[:, 0]).tolist()) == [0.0, 10.0]


def test_kmeans_con_menos_colores_distintos_que_k():
    X = np.vstack([np.zeros((50, 3)), np.full((50, 3), 9.0)])
    etiquetas, centros, _ = kmeans(X, 4, seed=0)
    assert not np.isnan(centros).any() and len(set(etiquetas[:50])) == 1 and etiquetas[0] != etiquetas[-1]


def test_componentes_por_etiqueta_ignora_barreras():
    et = np.array([[0, 0, 255, 1], [0, 255, 255, 1]], dtype=np.uint8)
    comp, n = componentes_por_etiqueta(et)
    assert n == 2 and comp[0, 0] == comp[1, 0] and comp[0, 3] == comp[1, 3] and comp[0, 2] == -1


def test_rellenar_huecos_y_cierre():
    m = np.zeros((20, 20), dtype=bool)
    m[2:18, 2:18] = True
    m[8:11, 8:11] = False
    assert rellenar_huecos(m)[9, 9]
    partida = np.zeros((20, 40), dtype=bool)
    partida[5:15, 2:18] = True
    partida[5:15, 22:38] = True
    assert cerrar_caja(partida, 3)[10, 20]          # un hueco de 4 px se cierra
    assert not partida[10, 20]


def test_segmenta_prenda_de_color_liso():
    img, gt = foto_sintetica()
    solo_color = segmentar_imagen(img, SegmentacionConfig(peso_textura=0))
    assert solo_color.encontrada and solo_color.mascara.shape == gt.shape
    assert (solo_color.mascara & gt).sum() / (solo_color.mascara | gt).sum() > 0.97
    # Con textura el borde forma su propio cluster y la máscara engorda unos píxeles (sesgo conocido)
    con_textura = segmentar_imagen(img, SegmentacionConfig(refinar_otsu=False))
    assert (con_textura.mascara & gt).sum() == gt.sum()          # no pierde nada de la prenda
    assert (con_textura.mascara & gt).sum() / (con_textura.mascara | gt).sum() > 0.85


def test_prenda_que_toca_el_borde_inferior():
    img, gt = foto_sintetica()
    img[170:, 40:120] = (180, 40, 40)
    gt[170:, 40:120] = True
    res = segmentar_imagen(img, SegmentacionConfig(peso_textura=0))
    assert res.encontrada
    assert (res.mascara & gt).sum() / (res.mascara | gt).sum() > 0.9


def test_conserva_el_espacio_entre_las_piernas():
    img = np.full((240, 160, 3), 235, dtype=np.uint8)
    img[20:230, 30:75] = (30, 50, 110)     # pierna izquierda
    img[20:230, 90:135] = (30, 50, 110)    # pierna derecha
    img[20:60, 30:135] = (30, 50, 110)     # cintura
    res = segmentar_imagen(img, SegmentacionConfig(peso_textura=0))
    assert res.encontrada
    assert not res.mascara[150, 82]        # el hueco entre piernas sigue siendo fondo
    assert res.mascara[150, 50] and res.mascara[150, 110]


def test_misma_semilla_misma_mascara():
    img, _ = foto_sintetica()
    a = segmentar_imagen(img, SegmentacionConfig(semilla=5)).mascara
    b = segmentar_imagen(img, SegmentacionConfig(semilla=5)).mascara
    assert np.array_equal(a, b)


def test_recortar_fondo_deja_transparente_lo_que_no_es_prenda():
    img, gt = foto_sintetica()
    rgba = recortar_fondo(img, gt)
    assert rgba.shape[2] == 4 and rgba[0, 0, 3] == 0 and rgba[100, 80, 3] == 255 and tuple(rgba[0, 0, :3]) == (0, 0, 0)


def test_con_otsu_por_defecto_el_borde_mejora_en_el_caso_sintetico():
    img, gt = foto_sintetica()
    sin = segmentar_imagen(img, SegmentacionConfig(refinar_otsu=False)).mascara
    con = segmentar_imagen(img).mascara                  # refinar_otsu=True por defecto
    iou = lambda m: (m & gt).sum() / (m | gt).sum()
    assert iou(con) > iou(sin)

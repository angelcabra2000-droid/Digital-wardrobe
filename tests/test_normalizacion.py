import numpy as np
import pytest

from closet.config import NormalizacionConfig
from closet.normalizacion import angulo_de_giro, momentos, muestrear_bilineal, normalizar_prenda


def rectangulo(alto_r=120, ancho_r=40, grados=0.0, H=300, W=300, centro=(150, 150)):
    yy, xx = np.mgrid[0:H, 0:W]
    a = np.radians(grados)
    u = (xx - centro[0]) * np.cos(a) + (yy - centro[1]) * np.sin(a)
    v = -(xx - centro[0]) * np.sin(a) + (yy - centro[1]) * np.cos(a)
    m = (abs(u) < ancho_r / 2) & (abs(v) < alto_r / 2)
    img = np.full((H, W, 3), 200, np.uint8)
    img[m] = (200, 30, 30)
    return img, m


def caja(m):
    ys, xs = np.nonzero(m)
    return xs.min(), xs.max(), ys.min(), ys.max()


def test_momentos_de_un_rectangulo_vertical():
    _, m = rectangulo()
    d = momentos(m)
    assert d["cx"] == pytest.approx(149.5, abs=0.6) and d["elongacion"] == pytest.approx(3.0, abs=0.1)
    assert abs(abs(np.degrees(d["theta"])) - 90) < 1


def test_endereza_un_rectangulo_inclinado_y_conserva_su_proporcion():
    img, m = rectangulo(grados=25)
    r = normalizar_prenda(img, m)
    x0, x1, y0, y1 = caja(r.mascara)
    assert (x1 - x0 + 1) < 0.45 * (y1 - y0 + 1)             # quedó alto y angosto, no inclinado
    assert r.meta["ratio_aspecto"] == pytest.approx(3.0, abs=0.15)
    assert abs(r.meta["giro_grados"]) == pytest.approx(25, abs=1.5)


def test_la_prenda_queda_centrada_y_con_margen():
    img, m = rectangulo(grados=0, centro=(90, 200))
    r = normalizar_prenda(img, m, NormalizacionConfig(lado=128))
    x0, x1, y0, y1 = caja(r.mascara)
    assert r.rgb.shape == (128, 128, 3) and r.mascara.shape == (128, 128)
    assert abs((x0 + x1) / 2 - 63.5) <= 1.5 and abs((y0 + y1) / 2 - 63.5) <= 1.5
    assert y0 >= 128 * 0.06 - 1 and y1 <= 128 * 0.94 + 1


def test_misma_prenda_en_otra_posicion_da_el_mismo_resultado():
    a = normalizar_prenda(*rectangulo(centro=(100, 120)))
    b = normalizar_prenda(*rectangulo(centro=(200, 180)))
    assert (a.mascara & b.mascara).sum() / (a.mascara | b.mascara).sum() > 0.97


def test_el_color_se_toma_de_la_original_y_fuera_de_la_mascara_es_cero():
    img, m = rectangulo()
    r = normalizar_prenda(img, m)
    assert tuple(r.rgb[r.nucleo][0]) == (200, 30, 30)
    assert not r.rgb[~r.mascara].any()
    assert r.nucleo.sum() < r.mascara.sum() and not (r.nucleo & ~r.mascara).any()


def test_no_endereza_prendas_poco_alargadas_ni_giros_ambiguos():
    cfg = NormalizacionConfig()
    cuadrada = {"theta": np.radians(10), "elongacion": 1.1}
    assert angulo_de_giro(cuadrada, cfg) == 0.0
    horizontal = {"theta": 0.0, "elongacion": 2.0}          # eje mayor horizontal: giro de 90°, ambiguo
    assert angulo_de_giro(horizontal, cfg) == 0.0
    casi_recta = {"theta": np.radians(89), "elongacion": 2.0}  # giro de 1°: no vale la pena
    assert angulo_de_giro(casi_recta, cfg) == 0.0
    assert angulo_de_giro({"theta": np.radians(60), "elongacion": 2.0}, cfg) == pytest.approx(np.radians(-30))
    assert angulo_de_giro({"theta": np.radians(60), "elongacion": 2.0}, NormalizacionConfig(enderezar=False)) == 0.0


def test_mascara_vacia_da_error():
    with pytest.raises(ValueError):
        normalizar_prenda(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool))


def test_muestreo_bilineal():
    img = np.array([[0, 100], [100, 200]], dtype=np.uint8)
    v = muestrear_bilineal(img, np.array([0.5, 0.0, 5.0]), np.array([0.5, 0.0, 5.0]))
    assert v[0] == pytest.approx(100) and v[1] == pytest.approx(0) and v[2] == 0   # fuera de la imagen: 0

import numpy as np

from closet.calidad import evaluar_calidad, varianza_laplaciano
from closet.preprocesamiento import suavizar_gaussiano


def _ruido(alto=400, ancho=400, seed=0):
    rng = np.random.default_rng(seed)
    return np.repeat(rng.integers(0, 256, (alto, ancho, 1)), 3, axis=2).astype(np.uint8)


def test_imagen_borrosa_tiene_menos_nitidez_y_se_marca():
    nitida = _ruido()
    borrosa = suavizar_gaussiano(nitida, 6.0)
    assert varianza_laplaciano(borrosa.mean(2)) < 0.01 * varianza_laplaciano(nitida.mean(2))
    assert "borrosa" not in evaluar_calidad(nitida)["advertencias"]
    assert "borrosa" in evaluar_calidad(borrosa)["advertencias"]


def test_imagen_oscura_y_sin_contraste():
    img = np.full((400, 400, 3), 10, dtype=np.uint8)
    adv = evaluar_calidad(img)["advertencias"]
    assert "oscura" in adv and "bajo_contraste" in adv


def test_imagen_sobreexpuesta():
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    assert "sobreexpuesta" in evaluar_calidad(img)["advertencias"]


def test_baja_resolucion():
    assert "baja_resolucion" in evaluar_calidad(_ruido(100, 100))["advertencias"]
    assert "baja_resolucion" not in evaluar_calidad(_ruido(400, 400))["advertencias"]


def test_metricas_en_rangos_validos():
    m = evaluar_calidad(_ruido())
    assert 0 <= m["brillo"] <= 255 and 0 <= m["frac_saturada"] <= 1 and m["alto"] == m["ancho"] == 400

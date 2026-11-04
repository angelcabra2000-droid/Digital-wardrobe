import numpy as np

from closet.config import PreprocesoConfig
from closet.preprocesamiento import (
    balance_blancos_mundo_gris,
    estirar_contraste,
    nucleo_gaussiano,
    preprocesar_imagen,
    promediar_bloques,
    redimensionar_lado_max,
    remuestrear_bilineal,
    suavizar_gaussiano,
)


def _ruido_neutro(alto=120, ancho=160, seed=0):
    """Imagen gris con variación: promedio ~128 en los tres canales."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 35, (alto, ancho, 1))
    return np.repeat(base, 3, axis=2).clip(0, 255)


# ------------------------------------------------------------------ tamaño
def test_redimensionar_conserva_proporcion():
    img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    assert redimensionar_lado_max(img, 800).shape == (400, 800, 3)


def test_redimensionar_no_amplia():
    img = np.zeros((300, 200, 3), dtype=np.uint8)
    assert redimensionar_lado_max(img, 800).shape == (300, 200, 3)


def test_redimensionar_color_constante_se_mantiene():
    img = np.full((1500, 1100, 3), (30, 90, 200), dtype=np.uint8)
    out = redimensionar_lado_max(img, 800)
    assert (out == (30, 90, 200)).all()


def test_promediar_bloques():
    img = np.array([[[0], [10]], [[20], [30]]], dtype=np.uint8)
    assert promediar_bloques(img, 2)[0, 0, 0] == 15


def test_bilineal_gradiente_horizontal_es_monotono():
    fila = np.linspace(0, 255, 400).astype(np.uint8)
    img = np.repeat(fila[None, :, None], 50, axis=0).repeat(3, axis=2)
    out = remuestrear_bilineal(img, 25, 100)
    perfil = out[0, :, 0].astype(int)
    assert out.shape == (25, 100, 3)
    assert (np.diff(perfil) >= 0).all()
    assert perfil[0] <= 5 and perfil[-1] >= 250


# ------------------------------------------------------------------- color
def test_balance_corrige_un_tinte_calido():
    neutra = _ruido_neutro()
    tinte = np.array([1.10, 1.0, 0.90])
    img = (neutra * tinte).clip(0, 255).astype(np.uint8)

    corregida, ganancias = balance_blancos_mundo_gris(img, ganancia_max=1.3)

    antes = img.reshape(-1, 3).mean(0)
    despues = corregida.reshape(-1, 3).mean(0)
    assert (antes.max() - antes.min()) / antes.mean() > 0.12
    assert (despues.max() - despues.min()) / despues.mean() < 0.03
    assert ganancias[0] < 1.0 < ganancias[2]


def test_balance_respeta_el_limite_de_ganancia():
    img = (_ruido_neutro() * np.array([1.6, 1.0, 0.6])).clip(0, 255).astype(np.uint8)
    _, ganancias = balance_blancos_mundo_gris(img, ganancia_max=1.15)
    assert all(1 / 1.15 - 1e-6 <= g <= 1.15 + 1e-6 for g in ganancias)


def test_balance_imagen_negra_no_falla():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    out, ganancias = balance_blancos_mundo_gris(img)
    assert (out == 0).all() and ganancias == (1.0, 1.0, 1.0)


def test_contraste_expande_un_rango_estrecho():
    rng = np.random.default_rng(2)
    img = np.repeat(rng.integers(100, 141, (80, 80, 1)), 3, axis=2).astype(np.uint8)
    out, _ = estirar_contraste(img, expansion_max=10.0)
    assert out.dtype == np.uint8
    assert out.std() > 2.5 * img.std()
    assert out.min() <= 5 and out.max() >= 250


def test_contraste_respeta_la_expansion_maxima():
    rng = np.random.default_rng(3)
    img = np.repeat(rng.integers(100, 141, (80, 80, 1)), 3, axis=2).astype(np.uint8)
    out, _ = estirar_contraste(img, expansion_max=1.5)
    assert out.std() <= 1.5 * img.std() + 1.0


def test_contraste_imagen_plana_no_falla():
    img = np.full((10, 10, 3), 77, dtype=np.uint8)
    out, _ = estirar_contraste(img)
    assert (out == 77).all()


# ---------------------------------------------------------------- suavizado
def test_nucleo_gaussiano_suma_uno_y_es_simetrico():
    k = nucleo_gaussiano(1.5)
    assert abs(k.sum() - 1.0) < 1e-6
    assert np.allclose(k, k[::-1])


def test_suavizado_reduce_ruido_y_conserva_constantes():
    ruido = _ruido_neutro().astype(np.uint8)
    assert suavizar_gaussiano(ruido, 1.5).std() < 0.6 * ruido.std()
    plana = np.full((30, 30, 3), 90, dtype=np.uint8)
    assert (suavizar_gaussiano(plana, 2.0) == 90).all()


def test_suavizado_sigma_cero_no_cambia_nada():
    img = _ruido_neutro().astype(np.uint8)
    assert np.array_equal(suavizar_gaussiano(img, 0), img)


# ----------------------------------------------------------------- pipeline
def test_pipeline_completo_formas_y_tipos():
    img = _ruido_neutro(1200, 900).astype(np.uint8)
    res = preprocesar_imagen(img, PreprocesoConfig(lado_max=600))
    assert res.original.shape == res.normalizada.shape == (600, 450, 3)
    assert res.original.dtype == res.normalizada.dtype == np.uint8
    assert {"nitidez", "brillo", "advertencias"} <= set(res.calidad)
    assert "ganancias_rgb" in res.pasos and "rango_luminancia" in res.pasos


def test_pipeline_no_modifica_la_entrada_ni_la_original():
    img = _ruido_neutro(200, 200).astype(np.uint8)
    copia = img.copy()
    res = preprocesar_imagen(img)
    assert np.array_equal(img, copia)
    assert np.array_equal(res.original, copia)  # original = sin corregir color

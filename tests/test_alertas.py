import numpy as np
from PIL import Image

from closet.alertas import fondo_impide_segmentar, revisar_segmentacion
from closet.pipeline import procesar_foto


def escena(color_prenda=(180, 40, 40), color_fondo=(235, 235, 235), area=(slice(40, 170), slice(40, 120))):
    img = np.empty((200, 160, 3), np.uint8)
    img[:] = color_fondo
    img[area] = color_prenda
    m = np.zeros((200, 160), bool)
    m[area] = True
    return img, m


def test_segmentacion_limpia_no_genera_alertas():
    img, m = escena()
    alertas, met = revisar_segmentacion(img, m)
    assert alertas == [] and met["contraste"] > 60 and met["fondo_sigma"] < 1


def test_filtro_de_fondo_requiere_dos_indicios():
    assert not fondo_impide_segmentar(['fondo_complejo'])
    assert not fondo_impide_segmentar(['prenda_parecida_fondo'])
    assert fondo_impide_segmentar(['fondo_complejo', 'prenda_parecida_fondo'])


def test_filtro_detecta_mascara_pegada_a_dos_bordes():
    img, mask = escena(area=(slice(0, 100), slice(0, 80)))
    alerts, metrics = revisar_segmentacion(img, mask)
    assert metrics['contacto_bordes']['superior'] == .5
    assert metrics['contacto_bordes']['izquierdo'] == .5
    assert fondo_impide_segmentar(alerts)


def test_prenda_del_color_del_fondo():
    img, m = escena(color_prenda=(225, 225, 228))
    assert "prenda_parecida_fondo" in revisar_segmentacion(img, m)[0]


def test_fondo_estampado():
    img, m = escena()
    rng = np.random.default_rng(0)
    ruido = rng.integers(0, 255, img.shape, dtype=np.uint8)
    img = np.where(m[..., None], img, ruido)
    assert "fondo_complejo" in revisar_segmentacion(img, m)[0]


def test_prenda_gigante_y_prenda_diminuta():
    img, m = escena(area=(slice(2, 198), slice(2, 158)))
    assert "prenda_muy_grande" in revisar_segmentacion(img, m)[0]
    img, m = escena(area=(slice(90, 100), slice(70, 80)))
    assert "prenda_muy_chica" in revisar_segmentacion(img, m)[0]


def test_pipeline_de_punta_a_punta(tmp_path):
    img, _ = escena()
    ruta = tmp_path / "foto.png"
    Image.fromarray(img).save(ruta)
    r = procesar_foto(ruta)
    assert r.normalizada is not None and r.alertas == []
    assert r.normalizada.mascara.any() and r.metricas["ratio_aspecto"] > 1.4


def test_foto_vacia_no_pasa_sin_alerta(tmp_path):
    """Una foto lisa sin prenda no puede salir como si estuviera bien: o no hay prenda o es 'muy grande'."""
    Image.fromarray(np.full((120, 120, 3), 128, np.uint8)).save(tmp_path / "vacia.png")
    r = procesar_foto(tmp_path / "vacia.png")
    assert r.alertas and (r.normalizada is None or "prenda_muy_grande" in r.alertas)

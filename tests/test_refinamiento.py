import numpy as np

from closet.config import SegmentacionConfig
from closet.refinamiento import refinar_con_otsu, umbral_otsu
from closet.segmentacion import dilatacion, erosion, segmentar_imagen


def foto(alto=200, ancho=160, fondo=(235, 235, 235), prenda=(180, 40, 40)):
    img = np.empty((alto, ancho, 3), np.uint8)
    img[:] = fondo
    img[40:170, 40:120] = prenda
    gt = np.zeros((alto, ancho), bool)
    gt[40:170, 40:120] = True
    return img, gt


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_otsu_separa_dos_grupos():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(10, 1, 500), rng.normal(50, 1, 500)])
    t, sep = umbral_otsu(v)
    assert 15 < t < 45 and sep > 0.9


def test_otsu_un_solo_grupo_tiene_separacion_baja():
    v = np.random.default_rng(1).normal(0, 1, 2000)
    assert umbral_otsu(v)[1] < 0.75
    assert umbral_otsu(np.full(100, 3.0))[1] == 0.0


def test_corrige_una_mascara_gorda_y_una_flaca():
    img, gt = foto()
    cfg = SegmentacionConfig(banda_otsu=0.03)
    gorda, flaca = dilatacion(gt, 3), erosion(gt, 3)
    for mala in (gorda, flaca):
        nueva, diag = refinar_con_otsu(img, mala, cfg)
        assert diag["aplicado"], diag
        assert iou(nueva, gt) > iou(mala, gt) and iou(nueva, gt) > 0.98


def test_no_cambia_nada_si_la_mascara_ya_es_exacta():
    img, gt = foto()
    nueva, diag = refinar_con_otsu(img, gt, SegmentacionConfig(banda_otsu=0.03))
    assert iou(nueva, gt) > 0.99


def test_prenda_y_fondo_iguales_deja_la_mascara_como_estaba():
    img, gt = foto(prenda=(235, 235, 235))
    img = np.clip(img.astype(int) + np.random.default_rng(2).integers(-2, 3, img.shape), 0, 255).astype(np.uint8)
    nueva, diag = refinar_con_otsu(img, gt, SegmentacionConfig(banda_otsu=0.03))
    assert not diag["aplicado"] and np.array_equal(nueva, gt)


def test_prenda_rayada_no_se_come_las_rayas():
    img, gt = foto()
    img[40:170, 60:70] = (240, 240, 240)       # raya clara del interior (no toca el borde)
    mala = dilatacion(gt, 3)
    nueva, _ = refinar_con_otsu(img, mala, SegmentacionConfig(banda_otsu=0.03))
    assert nueva[100, 65]                       # la raya sigue siendo prenda (está en el interior seguro)
    assert iou(nueva, gt) > iou(mala, gt)


def test_se_puede_apagar_y_encender_en_el_pipeline():
    img, gt = foto()
    base = segmentar_imagen(img, SegmentacionConfig(peso_textura=0, refinar_otsu=False))
    assert "otsu" not in base.diagnostico
    ref = segmentar_imagen(img, SegmentacionConfig(peso_textura=0, refinar_otsu=True))
    assert "otsu" in ref.diagnostico and ref.encontrada
    assert iou(ref.mascara, gt) >= iou(base.mascara, gt) - 0.01


def test_la_mascara_refinada_es_deterministica():
    img, gt = foto()
    m = dilatacion(gt, 3)
    a, _ = refinar_con_otsu(img, m)
    b, _ = refinar_con_otsu(img, m)
    assert np.array_equal(a, b)

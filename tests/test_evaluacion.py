import numpy as np
import pytest
from PIL import Image

from closet.dataset import Registro
from closet.evaluacion import buscar_pares, cargar_mascara, metricas, reducir_mascara, resumen, superposicion, Medicion, Par


def test_metricas_perfecta_y_nula():
    a = np.zeros((10, 10), dtype=bool)
    a[2:8, 2:8] = True
    assert metricas(a, a) == (1.0, 1.0, 1.0, 1.0)
    assert metricas(None, a) == (0.0, 0.0, 0.0, 0.0)
    assert metricas(np.zeros_like(a), a) == (0.0, 0.0, 0.0, 0.0)


def test_metricas_parcial():
    gt = np.zeros((10, 10), dtype=bool)
    gt[:, :5] = True                       # 50 px
    pred = np.zeros_like(gt)
    pred[:, :3] = True                     # 30 px, todos dentro
    iou, prec, rec, f1 = metricas(pred, gt)
    assert iou == pytest.approx(0.6) and prec == pytest.approx(1.0) and rec == pytest.approx(0.6)
    assert f1 == pytest.approx(2 * 1.0 * 0.6 / 1.6)


def test_reducir_mascara_vecino_mas_cercano():
    gt = np.zeros((100, 200), dtype=bool)
    gt[:, 100:] = True
    r = reducir_mascara(gt, 10, 20)
    assert r.shape == (10, 20) and r[:, :10].sum() == 0 and r[:, 10:].all()
    assert reducir_mascara(gt, 100, 200) is gt


def test_buscar_pares_respeta_el_conjunto(tmp_path):
    def reg(id_, split, muestra=True):
        return Registro(id=id_, ruta=f"camisas/{id_}.jpg", categoria="camisas", ancho=1, alto=1, sha1=id_,
                        ahash="0", grupo=id_, incluir=True, split=split, muestra_mascara=muestra)
    regs = [reg("a", "entrenamiento"), reg("b", "prueba"), reg("c", "entrenamiento", muestra=False),
            reg("d", "entrenamiento")]
    for n in ("a", "b", "c"):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(tmp_path / f"camisas__{n}.jpg")
        Image.fromarray(np.zeros((4, 4), np.uint8)).save(tmp_path / f"camisas__{n}_gt.png")
    pares, faltan = buscar_pares(regs, tmp_path, "entrenamiento")
    assert [p.registro.id for p in pares] == ["a"]
    assert faltan == ["camisas__d.jpg"]
    assert cargar_mascara(pares[0].mascara).shape == (4, 4)


def test_resumen_agrupa_por_categoria():
    def med(cat, iou):
        r = Registro(id="x", ruta="x", categoria=cat, ancho=1, alto=1, sha1="x", ahash="0", grupo="x")
        return Medicion(Par(r, None, None), iou, 1, 1, 1, 0.5, 0.5, True)
    r = resumen([med("camisas", 0.9), med("camisas", 0.5), med("pantalones", 1.0)])
    assert r["camisas"]["n"] == 2 and r["camisas"]["iou"] == pytest.approx(0.7) and r["camisas"]["bajo_080"] == 1
    assert r["TOTAL"]["n"] == 3 and list(r)[-1] == "TOTAL"


def test_superposicion_colorea_aciertos_faltantes_y_sobrantes():
    rgb = np.full((1, 3, 3), 100, np.uint8)
    pred = np.array([[True, True, False]])
    gt = np.array([[False, True, True]])
    out = superposicion(rgb, pred, gt)
    assert out[0, 1, 1] > out[0, 1, 0] and out[0, 2, 0] > out[0, 2, 1] and out[0, 0, 2] > out[0, 0, 0]

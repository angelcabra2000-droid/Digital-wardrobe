import numpy as np
import pytest

from closet.caracteristicas import (DIRECCIONES, NOMBRES_HARALICK, area_casco, caracteristicas_color,
                                    caracteristicas_forma, caracteristicas_textura, columnas_forma,
                                    cuantizar, describir_estampado, extraer_caracteristicas, haralick,
                                    matriz_coocurrencia, momentos_hu, nombre_color)
from closet.config import CaracteristicasConfig
from closet.normalizacion import ResultadoNormalizacion
from closet.segmentacion import rgb_a_lab


def rect(alto, ancho, H=100, W=100, y0=None, x0=None):
    y0 = (H - alto) // 2 if y0 is None else y0
    x0 = (W - ancho) // 2 if x0 is None else x0
    m = np.zeros((H, W), bool)
    m[y0:y0 + alto, x0:x0 + ancho] = True
    return m


def pantalon(H=100, W=100):
    """Una "Π": cintura ancha arriba y dos piernas separadas."""
    m = rect(20, 60, H, W, y0=10, x0=20)
    m[30:90, 20:45] = True
    m[30:90, 55:80] = True
    return m


# ------------------------------------------------------------------ color
@pytest.mark.parametrize("rgb,esperado", [((200, 30, 40), "rojo"), ((18, 18, 18), "negro"),
                                           ((250, 250, 250), "blanco"), ((25, 35, 85), "azul marino")])
def test_nombre_color_de_colores_de_la_paleta(rgb, esperado):
    assert nombre_color(rgb_a_lab(np.array([rgb], float))[0]) == esperado


def test_color_de_una_prenda_de_un_solo_color():
    rgb = np.zeros((60, 60, 3), np.uint8)
    rgb[:] = (200, 30, 40)
    c = caracteristicas_color(rgb, np.ones((60, 60), bool))
    assert c["color_nombre"] == "rojo" and c["n_colores"] == 1 and c["color_frac"] == pytest.approx(1.0)
    assert (c["color_r"], c["color_g"], c["color_b"]) == (200, 30, 40) and c["color_hex"] == "#c81e28"


def test_color_ignora_lo_que_esta_fuera_del_nucleo():
    rgb = np.zeros((60, 60, 3), np.uint8)
    rgb[:] = (0, 200, 0)                                   # fondo verde
    nucleo = rect(30, 30, 60, 60)
    rgb[nucleo] = (30, 60, 200)
    c = caracteristicas_color(rgb, nucleo)
    assert c["color_nombre"] == "azul" and c["n_colores"] == 1


def test_dos_colores_distintos_se_cuentan_y_el_dominante_es_el_mas_grande():
    rgb = np.zeros((60, 60, 3), np.uint8)
    rgb[:, :40] = (240, 240, 240)
    rgb[:, 40:] = (20, 20, 20)
    c = caracteristicas_color(rgb, np.ones((60, 60), bool))
    assert c["n_colores"] == 2 and c["color_nombre"] == "blanco" and c["color2_nombre"] == "negro"
    assert c["n_colores_cromaticos"] == 2
    assert c["color_frac"] == pytest.approx(2 / 3, abs=0.02)


def test_composicion_cromatica_agrupa_las_sombras_y_distingue_tonos():
    rgb = np.full((60, 60, 3), (134, 110, 59), np.uint8)
    rgb[:, 30:] = (76, 59, 27)
    mask = np.ones(rgb.shape[:2], bool)
    colors = caracteristicas_color(rgb, mask)
    assert colors['n_colores'] == 2 and colors['n_colores_cromaticos'] == 1
    rgb[:, 30:] = (30, 60, 150)
    assert caracteristicas_color(rgb, mask)['n_colores_cromaticos'] == 2


def test_color_con_menos_colores_que_k_no_falla():
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[:] = (10, 10, 10)
    c = caracteristicas_color(rgb, np.ones((20, 20), bool), CaracteristicasConfig(k_color=5))
    assert c["n_colores"] == 1


def test_color_con_nucleo_vacio_da_error():
    with pytest.raises(ValueError):
        caracteristicas_color(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool))


# ------------------------------------------------------------------ forma
def test_hu_no_cambia_al_mover_ni_al_escalar_ni_al_girar_90():
    a = pantalon()
    b = np.zeros((300, 300), bool)
    b[20:20 + 80 * 2, 40:40 + 60 * 2] = np.kron(a[10:90, 20:80], np.ones((2, 2), bool))   # x2 y movida
    c = np.rot90(a)
    h = momentos_hu(a)
    assert momentos_hu(b)[:4] == pytest.approx(h[:4], abs=0.05)
    assert momentos_hu(c)[:4] == pytest.approx(h[:4], abs=0.05)


def test_hu_distingue_formas():
    assert abs(momentos_hu(rect(60, 60))[1] - momentos_hu(pantalon())[1]) > 0.3


def test_area_casco_de_un_rectangulo_es_su_area():
    assert area_casco(rect(30, 50)) == pytest.approx(30 * 50)


def test_area_casco_de_un_circulo_es_cercana_al_area():      # el casco cuenta el píxel completo: ~3 % de más
    yy, xx = np.mgrid[0:100, 0:100]
    c = (yy - 50) ** 2 + (xx - 50) ** 2 <= 30 ** 2
    assert c.sum() <= area_casco(c) == pytest.approx(c.sum(), rel=0.05)


def test_solidez_de_forma_en_L_es_menor_que_1():
    m = rect(80, 20, 100, 100, y0=10, x0=10)
    m[70:90, 10:90] = True
    f = caracteristicas_forma(m)
    assert 0.45 < f["solidez"] < 0.8
    assert caracteristicas_forma(rect(40, 40))["solidez"] == pytest.approx(1.0)


def test_proporciones_de_un_rectangulo():
    f = caracteristicas_forma(rect(60, 30))
    assert f["ratio_aspecto"] == pytest.approx(2.0) and f["extension"] == pytest.approx(1.0)
    assert f["simetria"] == pytest.approx(1.0) and f["perimetro_rel"] == pytest.approx(1.0)
    assert f["fraccion_superior"] == pytest.approx(0.5)


def test_simetria_baja_si_la_prenda_es_asimetrica():
    m = rect(40, 20)
    m[10:30, 40:70] = True
    assert caracteristicas_forma(m)["simetria"] < 0.8


def test_un_pantalon_tiene_un_tramo_arriba_y_dos_abajo():
    f = caracteristicas_forma(pantalon())
    assert f["tramos_0"] == 1 and f["tramos_9"] == 2 and f["tramos_8"] == 2
    assert f["anchura_9"] == pytest.approx(60 / 60, abs=0.01) and f["relleno_9"] < 0.85   # hueco entre piernas


def test_un_rectangulo_siempre_tiene_un_tramo_y_anchura_1():
    f = caracteristicas_forma(rect(50, 40))
    assert all(f[f"tramos_{i}"] == 1 and f[f"anchura_{i}"] == pytest.approx(1.0) for i in range(10))


def test_tramos_muy_finos_se_toman_como_ruido():
    m = rect(40, 60)
    m[:, 59] = False
    m[20, 70] = True                                         # un píxel suelto a la derecha, fuera de la caja principal
    assert caracteristicas_forma(m)["tramos_5"] >= 1


def test_forma_con_mascara_vacia_da_error():
    with pytest.raises(ValueError):
        caracteristicas_forma(np.zeros((10, 10), bool))


def test_columnas_forma_existen_todas_y_son_numericas():
    f = caracteristicas_forma(pantalon())
    cols = columnas_forma()
    assert all(c in f for c in cols) and all(isinstance(f[c], (int, float)) for c in cols)


# ---------------------------------------------------------------- textura
def glcm_ingenua(q, valido, dy, dx, niveles):
    P = np.zeros((niveles, niveles))
    for y in range(q.shape[0]):
        for x in range(q.shape[1]):
            y2, x2 = y + dy, x + dx
            if 0 <= y2 < q.shape[0] and 0 <= x2 < q.shape[1] and valido[y, x] and valido[y2, x2]:
                P[q[y, x], q[y2, x2]] += 1
                P[q[y2, x2], q[y, x]] += 1
    return P / P.sum()


@pytest.mark.parametrize("angulo", [0, 45, 90, 135])
@pytest.mark.parametrize("distancia", [1, 3])
def test_glcm_coincide_con_una_version_con_bucles(angulo, distancia):
    rng = np.random.default_rng(angulo + distancia)
    q = rng.integers(0, 6, (14, 17))
    valido = rng.random((14, 17)) > 0.25
    dy, dx = (d * distancia for d in DIRECCIONES[angulo])
    P = matriz_coocurrencia(q, valido, dy, dx, 6)
    assert P == pytest.approx(glcm_ingenua(q, valido, dy, dx, 6))
    assert P.sum() == pytest.approx(1.0) and np.allclose(P, P.T)


def test_glcm_sin_pares_validos_da_none():
    q = np.zeros((5, 5), int)
    valido = np.zeros((5, 5), bool)
    valido[2, 2] = True
    assert matriz_coocurrencia(q, valido, 0, 1, 4) is None


def test_cuantizar_reparte_0_255_en_partes_iguales():
    q = cuantizar(np.array([0.0, 15.9, 16.0, 255.0]), 16)
    assert list(q) == [0, 0, 1, 15]


def haralick_ingenuo(P):
    N = P.shape[0]
    px, py = P.sum(1), P.sum(0)
    r = {"energia": 0.0, "contraste": 0.0, "homogeneidad": 0.0, "entropia": 0.0}
    mux = sum(i * px[i] for i in range(N))
    muy = sum(j * py[j] for j in range(N))
    sx = np.sqrt(sum((i - mux) ** 2 * px[i] for i in range(N)))
    sy = np.sqrt(sum((j - muy) ** 2 * py[j] for j in range(N)))
    cov = 0.0
    for i in range(N):
        for j in range(N):
            p = P[i, j]
            r["energia"] += p * p
            r["contraste"] += (i - j) ** 2 * p
            r["homogeneidad"] += p / (1 + (i - j) ** 2)
            r["entropia"] -= p * np.log(p) if p > 0 else 0
            cov += i * j * p
    r["correlacion"] = (cov - mux * muy) / (sx * sy)
    suma = np.zeros(2 * N - 1)
    for i in range(N):
        for j in range(N):
            suma[i + j] += P[i, j]
    r["suma_entropia"] = -sum(p * np.log(p) for p in suma if p > 0)
    return r


def test_haralick_coincide_con_las_formulas_escritas_con_bucles():
    rng = np.random.default_rng(3)
    q = rng.integers(0, 8, (30, 30))
    P = matriz_coocurrencia(q, np.ones((30, 30), bool), 0, 1, 8)
    h, ref = haralick(P), haralick_ingenuo(P)
    for k, v in ref.items():
        assert h[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
    assert set(h) == set(NOMBRES_HARALICK) and all(np.isfinite(list(h.values())))


def test_haralick_de_una_imagen_constante():
    P = matriz_coocurrencia(np.full((10, 10), 5), np.ones((10, 10), bool), 0, 1, 16)
    h = haralick(P)
    assert h["energia"] == pytest.approx(1.0) and h["contraste"] == pytest.approx(0.0)
    assert h["homogeneidad"] == pytest.approx(1.0) and h["entropia"] == pytest.approx(0.0)
    assert h["correlacion"] == pytest.approx(1.0)               # definida como 1 cuando no hay variación


def test_haralick_de_un_tablero_en_dos_niveles():
    q = (np.indices((12, 12)).sum(0) % 2) * 15
    P = matriz_coocurrencia(q, np.ones((12, 12), bool), 0, 1, 16)
    h = haralick(P)
    assert h["contraste"] == pytest.approx(225.0) and h["homogeneidad"] == pytest.approx(1 / 226)
    assert h["energia"] == pytest.approx(0.5)


def lienzo(patron, lado=128):
    rgb = np.zeros((lado, lado, 3), np.uint8)
    rgb[:] = 200
    yy, xx = np.indices((lado, lado))
    if patron == "rayas_verticales":
        rgb[(xx // 4) % 2 == 0] = 30
    elif patron == "rayas_horizontales":
        rgb[(yy // 4) % 2 == 0] = 30
    elif patron == "cuadros":
        rgb[((xx // 4) + (yy // 4)) % 2 == 0] = 30
    return rgb


def test_las_rayas_verticales_varian_en_horizontal_y_las_horizontales_en_vertical():
    nuc = np.ones((128, 128), bool)
    rv = caracteristicas_textura(lienzo("rayas_verticales"), nuc)
    rh = caracteristicas_textura(lienzo("rayas_horizontales"), nuc)
    assert rv["patron_anisotropia"] > 0.8 and rh["patron_anisotropia"] < -0.8
    assert rv["patron_contraste_h"] > 10 * rv["patron_contraste_v"]


def test_los_cuadros_varian_en_las_dos_direcciones():
    t = caracteristicas_textura(lienzo("cuadros"), np.ones((128, 128), bool))
    assert abs(t["patron_anisotropia"]) < 0.2 and t["patron_contraste_h"] > 5 and t["patron_contraste_v"] > 5


def test_el_fondo_fuera_del_nucleo_no_entra_en_la_textura():
    rgb = lienzo("liso")
    rgb[:, :40] = 0                                            # franja negra (como el lienzo vacío del paso 3)
    nuc = np.ones((128, 128), bool)
    nuc[:, :40] = False
    t = caracteristicas_textura(rgb, nuc)
    assert t["har_contraste"] == pytest.approx(0.0) and t["har_energia"] == pytest.approx(1.0)


def test_textura_de_una_prenda_diminuta_da_error():
    nuc = np.zeros((20, 20), bool)
    nuc[5, 5] = True
    with pytest.raises(ValueError):
        caracteristicas_textura(np.zeros((20, 20, 3), np.uint8), nuc)


def test_estampado_liso_o_con_textura():
    nuc = np.ones((128, 128), bool)
    assert describir_estampado(caracteristicas_textura(lienzo("liso"), nuc)) == "liso"
    for patron in ("rayas_verticales", "rayas_horizontales", "cuadros"):
        assert describir_estampado(caracteristicas_textura(lienzo(patron), nuc)) == "con_textura", patron


def test_el_ruido_leve_de_una_tela_lisa_sigue_siendo_liso():
    rgb = np.clip(lienzo("liso").astype(int) + np.random.default_rng(0).integers(-3, 4, (128, 128, 1)), 0, 255)
    t = caracteristicas_textura(rgb.astype(np.uint8), np.ones((128, 128), bool))
    assert describir_estampado(t) == "liso"


# ---------------------------------------------------------------- conjunto
def prenda_sintetica():
    m = pantalon(256, 256)
    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[m] = (40, 70, 140)
    nucleo = m.copy()
    nucleo[:, [20, 79]] = False
    return ResultadoNormalizacion(rgb, m, nucleo, {})


def test_extraer_caracteristicas_de_punta_a_punta():
    c = extraer_caracteristicas(prenda_sintetica())
    d = c.como_dict()
    assert d["color_nombre"] in {"azul jean", "azul", "azul marino"} and d["estampado"] == "liso"
    assert d["tramos_9"] == 2 and d["ratio_aspecto"] > 1
    assert len(set(c.color) & set(c.forma) | set(c.color) & set(c.textura) | set(c.forma) & set(c.textura)) == 0


def test_extraer_caracteristicas_es_determinista():
    n = prenda_sintetica()
    assert extraer_caracteristicas(n).como_dict() == extraer_caracteristicas(n).como_dict()


def test_si_el_nucleo_esta_vacio_se_usa_la_mascara():
    n = prenda_sintetica()
    n.nucleo = np.zeros_like(n.nucleo)
    assert extraer_caracteristicas(n).color["color_nombre"] in {"azul jean", "azul", "azul marino"}


# ------------------------------------------- color y detalle sin sombras (ajustes con fotos reales)
@pytest.mark.parametrize("rgb,esperado", [((20, 30, 44), "azul marino"),     # negro con tinte azul (camiseta marino)
                                           ((45, 35, 33), "marrón"),           # pantalón marrón muy oscuro
                                           ((208, 205, 198), "blanco"),        # camiseta blanca con luz de cuarto
                                           ((30, 30, 30), "negro")])
def test_nombre_color_reglas_para_oscuros_y_blancos(rgb, esperado):
    assert nombre_color(rgb_a_lab(np.array([rgb], float))[0]) == esperado


def test_pliegues_suaves_no_cuentan_como_estampado():
    from closet.caracteristicas import detalle_sin_sombras
    y, x = np.mgrid[0:128, 0:128]
    sombra = 120 + 40 * np.sin(x / 14.0) * np.sin(y / 18.0)          # pliegues de baja frecuencia
    rgb = np.repeat(sombra[..., None], 3, axis=2).astype(np.uint8)
    assert detalle_sin_sombras(rgb, np.ones((128, 128), bool)) < 0.07


def test_estampado_claro_sobre_prenda_oscura_si_cuenta():
    rgb = np.full((128, 128, 3), 18, np.uint8)
    rgb[40:70, 30:100] = (170, 170, 175)                              # logo plateado sobre negro (~13 % del área)
    nuc = np.ones((128, 128), bool)
    t = caracteristicas_textura(rgb, nuc)
    assert describir_estampado(t) == "con_textura" and t["detalle_frac"] > 0.07

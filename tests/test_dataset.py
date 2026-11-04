import numpy as np
import pytest
from PIL import Image

from closet.cli import main
from closet.config import DatasetConfig
from closet.dataset import (
    ENTRENAMIENTO, PRUEBA, MOTIVO_MANUAL, Registro, aplicar_exclusiones, asignar_ids, cargar_csv,
    construir_dataset, copiar_muestra, dividir, diferencia_firmas, distancia_hamming, elegir_muestra,
    escanear, firma_color, guardar_csv, hash_promedio, leer_exclusiones, marcar_duplicados,
    nombre_en_muestra, resumen,
)
from closet.revision import generar_revision


# ------------------------------------------------------------------ utilidades
def patron(semilla: int, bloque: int = 40) -> np.ndarray:
    """Imagen de 8 x 8 bloques de color al azar (320 x 320): cada semilla da una foto distinta."""
    celdas = np.random.default_rng(semilla).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    return np.kron(celdas, np.ones((bloque, bloque, 1), dtype=np.uint8))


def guardar(img: np.ndarray, ruta) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(ruta)


def crear_base(tmp_path, por_categoria=None):
    """Base de ejemplo: una subcarpeta por categoría con fotos todas distintas."""
    por_categoria = por_categoria or {"camisas": 10, "chaquetas": 10, "pantalones": 10}
    raiz = tmp_path / "base"
    semilla = 0
    for categoria, n in por_categoria.items():
        for i in range(n):
            semilla += 1
            guardar(patron(semilla), raiz / categoria / f"foto_{i:02d}.png")
    return raiz


# ----------------------------------------------------------------- huella visual
def test_hash_distingue_copias_de_fotos_distintas():
    base = patron(1)
    mas_clara = np.clip(base.astype(int) + 8, 0, 255).astype(np.uint8)
    reducida = base[::2, ::2]
    assert distancia_hamming(hash_promedio(base), hash_promedio(mas_clara)) <= 4
    assert distancia_hamming(hash_promedio(base), hash_promedio(reducida)) <= 4
    assert distancia_hamming(hash_promedio(base), hash_promedio(patron(2))) > 10


def test_distancia_hamming():
    assert distancia_hamming(0b1010, 0b1010) == 0
    assert distancia_hamming(0b1010, 0b0101) == 4


# ----------------------------------------------------------------------- escaneo
def test_escanear_categorias_y_tamano(tmp_path):
    raiz = crear_base(tmp_path, {"camisas": 2, "pantalones": 1})
    guardar(patron(99), raiz / "suelta.png")
    regs = {r.ruta: r for r in escanear(raiz)}
    assert regs["camisas/foto_00.png"].categoria == "camisas"
    assert regs["pantalones/foto_00.png"].categoria == "pantalones"
    assert regs["suelta.png"].categoria == "sin_categoria"
    assert (regs["camisas/foto_00.png"].ancho, regs["camisas/foto_00.png"].alto) == (320, 320)
    assert len(regs["camisas/foto_00.png"].ahash) == 16


def test_escanear_no_se_cae_con_un_archivo_danado(tmp_path):
    raiz = crear_base(tmp_path, {"camisas": 1})
    (raiz / "camisas" / "rota.jpg").write_text("esto no es una imagen")
    regs = {r.ruta: r for r in escanear(raiz)}
    assert regs["camisas/rota.jpg"].incluir is False
    assert regs["camisas/rota.jpg"].motivo.startswith("error")
    assert regs["camisas/foto_00.png"].incluir is True


def test_escanear_avisa_baja_resolucion(tmp_path):
    raiz = tmp_path / "base"
    guardar(patron(1, bloque=10), raiz / "camisas" / "chica.png")   # 80 x 80
    assert "baja_resolucion" in escanear(raiz)[0].avisos


def test_escanear_exige_una_carpeta(tmp_path):
    with pytest.raises(NotADirectoryError):
        escanear(tmp_path / "no_existe")


# -------------------------------------------------------------------- duplicados
def test_copia_exacta_se_excluye_y_avisa_si_cambia_de_categoria(tmp_path):
    raiz = crear_base(tmp_path, {"camisas": 2})
    (raiz / "chaquetas").mkdir()
    (raiz / "chaquetas" / "gemela.png").write_bytes((raiz / "camisas" / "foto_00.png").read_bytes())
    regs = escanear(raiz)
    marcar_duplicados(regs)
    por_ruta = {r.ruta: r for r in regs}
    assert por_ruta["camisas/foto_00.png"].incluir
    assert not por_ruta["chaquetas/gemela.png"].incluir
    assert por_ruta["chaquetas/gemela.png"].motivo == "duplicado_exacto_de:camisas/foto_00.png"
    assert "duplicado_en_otra_categoria" in por_ruta["camisas/foto_00.png"].avisos
    assert por_ruta["chaquetas/gemela.png"].grupo == ""


def test_casi_duplicados_comparten_grupo(tmp_path):
    raiz = crear_base(tmp_path, {"camisas": 2})
    guardar(patron(1)[::2, ::2], raiz / "camisas" / "foto_00_chica.png")   # misma foto, reducida
    regs = escanear(raiz)
    marcar_duplicados(regs)
    por_ruta = {r.ruta: r for r in regs}
    a, b, c = (por_ruta[r] for r in ("camisas/foto_00.png", "camisas/foto_00_chica.png", "camisas/foto_01.png"))
    assert a.grupo == b.grupo != c.grupo
    assert "casi_duplicado" in a.avisos and "casi_duplicado" in b.avisos
    assert "casi_duplicado" not in c.avisos


def camiseta(color, fondo=(245, 245, 245), estampado=None) -> np.ndarray:
    """Prenda de catálogo: siempre el mismo corte sobre fondo liso."""
    a = np.full((400, 300, 3), fondo, np.uint8)
    a[90:330, 70:230] = color
    a[90:170, 20:70] = color
    a[90:170, 230:280] = color
    a[60:90, 125:175] = fondo
    if estampado:
        a[150:230, 110:190] = estampado
    return a


def test_prendas_distintas_con_la_misma_silueta_no_se_agrupan(tmp_path):
    """Con solo la huella en gris, una camiseta negra, una azul y una roja del mismo corte se agrupaban."""
    raiz = tmp_path / "base"
    for nombre, img in {
        "negra": camiseta((20, 20, 20)), "azul": camiseta((20, 40, 120)), "roja": camiseta((170, 20, 20)),
        "negra_estampada": camiseta((20, 20, 20), estampado=(220, 200, 60)),
    }.items():
        guardar(img, raiz / "camisas" / f"{nombre}.png")
    guardar(camiseta((20, 20, 20))[::2, ::2], raiz / "camisas" / "negra_reducida.png")   # copia reescalada
    regs = escanear(raiz)
    assert distancia_hamming(int(regs[0].ahash, 16), int(regs[1].ahash, 16)) <= 4   # el hash solo, no alcanza
    marcar_duplicados(regs)
    grupo = {r.ruta.split("/")[1]: r.grupo for r in regs}
    assert grupo["negra.png"] == grupo["negra_reducida.png"]                       # la copia sí
    otras = [grupo[n] for n in ("azul.png", "roja.png", "negra_estampada.png")]
    assert len({grupo["negra.png"], *otras}) == 4                                  # el resto, grupos propios
    avisos = {r.ruta.split("/")[1]: r.avisos for r in regs}
    assert "casi_duplicado" in avisos["negra.png"] and "casi_duplicado" not in avisos["azul.png"]


def test_firma_de_color():
    base = camiseta((20, 40, 120))
    assert len(firma_color(base)) == 12 * 12 * 3
    assert diferencia_firmas(firma_color(base), firma_color(base[::2, ::2])) < 2
    assert diferencia_firmas(firma_color(base), firma_color(camiseta((170, 20, 20)))) > 10


def test_proporciones_distintas_no_son_copias(tmp_path):
    raiz = tmp_path / "base"
    guardar(camiseta((20, 20, 20)), raiz / "camisas" / "a.png")
    guardar(camiseta((20, 20, 20))[:, :200], raiz / "camisas" / "recortada.png")     # mismo color, otro encuadre
    regs = escanear(raiz)
    marcar_duplicados(regs)
    assert regs[0].grupo != regs[1].grupo


# ------------------------------------------------------------------------- ids
def test_ids_cortos_y_unicos(tmp_path):
    regs = construir_dataset(crear_base(tmp_path))
    ids = [r.id for r in regs]
    assert len(set(ids)) == len(ids)
    assert {r.id for r in regs if r.categoria == "camisas"} == {f"cam_{i:03d}" for i in range(1, 11)}
    assert all(r.id.startswith("pan_") for r in regs if r.categoria == "pantalones")


def test_prefijo_sin_tildes_ni_espacios():
    regs = [Registro(ruta="Ñandú Ropa/a.png", categoria="Ñandú Ropa", sha1="1")]
    asignar_ids(regs)
    assert regs[0].id == "nan_001"


# --------------------------------------------------------------------- división
def test_division_estratificada_y_sin_fugas(tmp_path):
    regs = construir_dataset(crear_base(tmp_path))
    for categoria in ("camisas", "chaquetas", "pantalones"):
        splits = [r.split for r in regs if r.categoria == categoria]
        assert splits.count(PRUEBA) == 3          # 30 % de 10
        assert splits.count(ENTRENAMIENTO) == 7


def test_casi_duplicados_caen_del_mismo_lado(tmp_path):
    raiz = crear_base(tmp_path)
    for i in range(10):   # una gemela reducida de cada camisa
        guardar(patron(i + 1)[::2, ::2], raiz / "camisas" / f"gemela_{i:02d}.png")
    regs = construir_dataset(raiz)
    lados_por_grupo = {}
    for r in regs:
        if r.incluir:
            lados_por_grupo.setdefault(r.grupo, set()).add(r.split)
    assert all(len(lados) == 1 for lados in lados_por_grupo.values())


def test_categoria_con_una_sola_foto_va_a_entrenamiento(tmp_path):
    regs = construir_dataset(crear_base(tmp_path, {"camisas": 6, "faldas": 1}))
    assert [r.split for r in regs if r.categoria == "faldas"] == [ENTRENAMIENTO]


def test_division_es_reproducible(tmp_path):
    raiz = crear_base(tmp_path)
    a = {r.ruta: r.split for r in construir_dataset(raiz)}
    b = {r.ruta: r.split for r in construir_dataset(raiz)}
    c = {r.ruta: r.split for r in construir_dataset(raiz, DatasetConfig(semilla="otra"))}
    assert a == b
    assert a != c


def test_las_excluidas_no_tienen_division(tmp_path):
    raiz = crear_base(tmp_path)
    regs = construir_dataset(raiz, exclusiones={"camisas/foto_03.png"})
    r = next(r for r in regs if r.ruta == "camisas/foto_03.png")
    assert (r.incluir, r.motivo, r.split, r.grupo) == (False, MOTIVO_MANUAL, "", "")


# ---------------------------------------------------------------- estabilidad
def test_volver_a_correr_conserva_ids_division_y_muestra(tmp_path):
    raiz = crear_base(tmp_path)
    csv_ruta = tmp_path / "datos" / "dataset.csv"
    primera = construir_dataset(raiz)
    guardar_csv(primera, csv_ruta)

    for i in range(4):   # llegan fotos nuevas
        guardar(patron(500 + i), raiz / "camisas" / f"nueva_{i}.png")
    segunda = construir_dataset(raiz, previos=cargar_csv(csv_ruta))

    antes = {r.ruta: (r.id, r.split, r.muestra_mascara) for r in primera}
    despues = {r.ruta: (r.id, r.split, r.muestra_mascara) for r in segunda}
    for ruta, valores in antes.items():
        assert despues[ruta] == valores
    nuevas = [r for r in segunda if r.ruta.startswith("camisas/nueva_")]
    assert {r.id for r in nuevas} == {"cam_011", "cam_012", "cam_013", "cam_014"}
    assert all(r.split in (ENTRENAMIENTO, PRUEBA) for r in nuevas)


def test_exclusion_hecha_a_mano_en_el_csv_se_conserva(tmp_path):
    raiz = crear_base(tmp_path)
    csv_ruta = tmp_path / "datos" / "dataset.csv"
    guardar_csv(construir_dataset(raiz), csv_ruta)

    previos = cargar_csv(csv_ruta)
    next(p for p in previos if p.ruta == "pantalones/foto_00.png").incluir = False   # editado en Excel
    guardar_csv(previos, csv_ruta)

    regs = construir_dataset(raiz, previos=cargar_csv(csv_ruta))
    r = next(r for r in regs if r.ruta == "pantalones/foto_00.png")
    assert not r.incluir and r.motivo == MOTIVO_MANUAL


def test_redividir_rehace_la_division_sobre_las_fotos_que_quedan(tmp_path):
    raiz = crear_base(tmp_path)
    inicial = construir_dataset(raiz)
    de_prueba = [r.ruta for r in inicial if r.categoria == "camisas" and r.split == PRUEBA]
    excluir = set(de_prueba[:2]) | {"camisas/foto_09.png"}      # se descartan 2 fotos de prueba y 1 más

    conservando = construir_dataset(raiz, previos=inicial, exclusiones=excluir)
    rehecha = construir_dataset(raiz, previos=inicial, exclusiones=excluir, conservar_division=False)

    def prueba_camisas(regs):
        return sum(r.split == PRUEBA for r in regs if r.categoria == "camisas")

    assert prueba_camisas(conservando) == 1        # se conserva lo asignado: la prueba quedó chica
    assert prueba_camisas(rehecha) == 2            # 30 % de las 7 que quedan, redondeado
    assert {r.ruta: r.id for r in rehecha} == {r.ruta: r.id for r in inicial}   # los ids no cambian


# --------------------------------------------------------------------- muestra
def test_muestra_equilibrada_y_solo_de_las_incluidas(tmp_path):
    raiz = crear_base(tmp_path, {"camisas": 20, "chaquetas": 20, "pantalones": 20})
    regs = construir_dataset(raiz, DatasetConfig(mascaras_entrenamiento=6, mascaras_prueba=3),
                             exclusiones={"camisas/foto_00.png"})
    elegidas = [r for r in regs if r.muestra_mascara]
    assert len(elegidas) == 9
    for categoria in ("camisas", "chaquetas", "pantalones"):
        assert sum(r.categoria == categoria and r.split == ENTRENAMIENTO for r in elegidas) == 2
        assert sum(r.categoria == categoria and r.split == PRUEBA for r in elegidas) == 1
    assert all(r.incluir for r in elegidas)


def test_pedir_una_muestra_mayor_conserva_la_anterior(tmp_path):
    raiz = crear_base(tmp_path, {"camisas": 20, "chaquetas": 20, "pantalones": 20})
    chica = construir_dataset(raiz, DatasetConfig(mascaras_entrenamiento=6, mascaras_prueba=3))
    ruta_csv = tmp_path / "d.csv"
    guardar_csv(chica, ruta_csv)

    grande = cargar_csv(ruta_csv)
    elegir_muestra(grande, DatasetConfig(mascaras_entrenamiento=12, mascaras_prueba=6))
    antes = {r.ruta for r in chica if r.muestra_mascara}
    despues = {r.ruta for r in grande if r.muestra_mascara}
    assert antes < despues and len(despues) == 18


def test_una_sola_foto_por_grupo_en_la_muestra(tmp_path):
    raiz = crear_base(tmp_path, {"camisas": 6})
    for i in range(6):
        guardar(patron(i + 1)[::2, ::2], raiz / "camisas" / f"gemela_{i}.png")
    regs = construir_dataset(raiz, DatasetConfig(mascaras_entrenamiento=20, mascaras_prueba=20))
    grupos = [r.grupo for r in regs if r.muestra_mascara]
    assert len(grupos) == len(set(grupos))


def test_copiar_muestra(tmp_path):
    raiz = crear_base(tmp_path)
    regs = construir_dataset(raiz, DatasetConfig(mascaras_entrenamiento=3, mascaras_prueba=3))
    destino = tmp_path / "muestra"
    nuevas, total = copiar_muestra(regs, raiz, destino)
    assert nuevas == total == 6
    r = next(r for r in regs if r.muestra_mascara)
    assert (destino / nombre_en_muestra(r)).exists()
    assert nombre_en_muestra(r) == f"{r.categoria}__{r.ruta.split('/')[-1]}"
    assert copiar_muestra(regs, raiz, destino) == (0, 6)    # no repite lo ya copiado


# ------------------------------------------------------------------------- CSV
def test_csv_ida_y_vuelta(tmp_path):
    regs = construir_dataset(crear_base(tmp_path), exclusiones={"camisas/foto_01.png"})
    guardar_csv(regs, tmp_path / "dataset.csv")
    assert cargar_csv(tmp_path / "dataset.csv") == regs


def test_csv_acepta_si_no_escritos_a_mano(tmp_path):
    ruta = tmp_path / "dataset.csv"
    ruta.write_text("id,ruta,categoria,incluir\ncam_001,camisas\\a.jpg,camisas,No\ncam_002,camisas/b.jpg,camisas,SI\n"
                    "cam_003,camisas/c.jpg,camisas,\n", encoding="utf-8")
    a, b, c = cargar_csv(ruta)
    assert (a.incluir, b.incluir, c.incluir) == (False, True, True)
    assert a.ruta == "camisas/a.jpg"      # las barras de Windows se normalizan


def test_leer_exclusiones(tmp_path):
    ruta = tmp_path / "exclusiones.txt"
    ruta.write_text("# comentario\n\ncamisas\\a.jpg\n  pantalones/b.jpg  \n", encoding="utf-8")
    assert leer_exclusiones(ruta) == {"camisas/a.jpg", "pantalones/b.jpg"}
    assert leer_exclusiones(tmp_path / "no_existe.txt") == set()


def test_aplicar_exclusiones():
    regs = [Registro(ruta="a.jpg"), Registro(ruta="b.jpg")]
    aplicar_exclusiones(regs, {"b.jpg"})
    assert [r.incluir for r in regs] == [True, False]


def test_resumen_tiene_total(tmp_path):
    texto = resumen(construir_dataset(crear_base(tmp_path)))
    assert "TOTAL" in texto and "camisas" in texto
    ultima = texto.splitlines()[-1].split()
    assert ultima[:3] == ["TOTAL", "30", "30"]


# -------------------------------------------------------------------- revisión
def test_hoja_de_revision(tmp_path):
    raiz = crear_base(tmp_path, {"camisas": 3, "pantalones": 2})
    regs = construir_dataset(raiz, exclusiones={"camisas/foto_01.png"})
    html = generar_revision(regs, raiz, tmp_path / "revision")
    texto = html.read_text(encoding="utf-8")
    assert "cam_001" in texto and "pan_002" in texto
    assert "EXCLUIDA" in texto                    # estilo de las excluidas
    assert len(list((tmp_path / "revision" / "miniaturas").glob("*.jpg"))) == 5   # incluye la excluida a mano
    assert '"excluida": true' in texto


# ------------------------------------------------------------------------- CLI
def test_cli_inventario_y_muestra(tmp_path, capsys):
    raiz = crear_base(tmp_path, {"camisas": 12, "chaquetas": 12, "pantalones": 12})
    datos, muestra = tmp_path / "datos", tmp_path / "muestra"
    assert main(["inventario", str(raiz), "--datos", str(datos), "--revision"]) == 0
    assert (datos / "dataset.csv").exists() and (datos / "revision" / "revision.html").exists()
    assert "TOTAL" in capsys.readouterr().out

    assert main(["muestra", str(raiz), "--datos", str(datos), "--salida", str(muestra),
                 "--n-entrenamiento", "6", "--n-prueba", "3"]) == 0
    assert len(list(muestra.glob("*.png"))) == 9
    assert sum(r.muestra_mascara for r in cargar_csv(datos / "dataset.csv")) == 9


def test_cli_aplica_exclusiones_txt(tmp_path):
    raiz = crear_base(tmp_path)
    datos = tmp_path / "datos"
    datos.mkdir()
    (datos / "exclusiones.txt").write_text("camisas/foto_02.png\n", encoding="utf-8")
    assert main(["inventario", str(raiz), "--datos", str(datos)]) == 0
    r = next(r for r in cargar_csv(datos / "dataset.csv") if r.ruta == "camisas/foto_02.png")
    assert (r.incluir, r.motivo) == (False, MOTIVO_MANUAL)
    assert main(["inventario", str(raiz), "--datos", str(datos), "--redividir"]) == 0


def test_cli_avisa_si_solo_hay_una_categoria(tmp_path, capsys):
    """Caso real: la base quedó anidada (base/base/camisas/...) y todo cayó en una sola categoría."""
    raiz = tmp_path / "base"
    interior = tmp_path / "base" / "base"
    for categoria in ("camisas", "pantalones"):
        for i in range(4):
            guardar(patron(hash((categoria, i)) % 1000), interior / categoria / f"f{i}.png")
    assert main(["inventario", str(raiz), "--datos", str(tmp_path / "d")]) == 0
    salida = capsys.readouterr().out
    assert "solo se detectó una categoría" in salida and "carpeta anidada" in salida
    assert main(["inventario", str(interior), "--datos", str(tmp_path / "d2")]) == 0
    assert "solo se detectó" not in capsys.readouterr().out


def test_cli_errores_claros(tmp_path, capsys):
    assert main(["inventario", str(tmp_path / "no_existe")]) == 1
    raiz = crear_base(tmp_path, {"camisas": 2})
    assert main(["inventario", str(raiz), "--datos", str(tmp_path / "d"), "--fraccion-prueba", "1.5"]) == 1
    assert main(["muestra", str(raiz), "--datos", str(tmp_path / "sin_inventario")]) == 1
    assert "Corre primero" in capsys.readouterr().err

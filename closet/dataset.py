"""Paso 0: inventario de la base de imágenes y su división en entrenamiento y prueba.

Por qué existe este paso: con 200 o más fotos hay que saber qué se tiene (categorías, fotos dañadas,
repetidas o muy pequeñas) y, sobre todo, **separar desde el principio** las fotos con las que se ajustan
los parámetros (entrenamiento) de las que solo se usan para medir al final (prueba). Si se ajusta mirando
todas las fotos, las métricas finales salen mejores de lo que realmente son.

Resultado: un `dataset.csv` con una fila por foto (ver `Registro`). Es un archivo normal: se puede abrir
en Excel y editar a mano (por ejemplo `incluir = no` para una foto con una persona puesta).

Reglas:
  - La categoría es la primera subcarpeta (camisas, chaquetas, pantalones...).
  - Los originales nunca se tocan ni se renombran: cada foto recibe un `id` estable (cam_001, pan_014...).
  - Duplicados exactos (mismo contenido): solo se conserva uno.
  - Casi duplicados (misma foto reescalada o recomprimida): se agrupan y **caen siempre del mismo lado**
    de la división, para que una foto de prueba no tenga una gemela en entrenamiento.
  - Volver a correr el inventario conserva ids, división y muestra de máscaras ya decididos.
"""
from __future__ import annotations

import csv
import hashlib
import random
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from .calidad import luminancia
from .config import CalidadConfig, DatasetConfig
from .imagen_io import cargar_imagen, listar_imagenes
from .preprocesamiento import redimensionar_lado_max, remuestrear_bilineal

ENTRENAMIENTO = "entrenamiento"
PRUEBA = "prueba"
SIN_CATEGORIA = "sin_categoria"
MOTIVO_MANUAL = "exclusion_manual"


@dataclass
class Registro:
    """Una foto de la base."""

    id: str = ""
    ruta: str = ""              # relativa a la carpeta de la base, siempre con "/"
    categoria: str = ""
    ancho: int = 0              # tamaño ya con la orientación EXIF corregida
    alto: int = 0
    sha1: str = ""              # huella del archivo (detecta copias idénticas)
    ahash: str = ""             # huella visual de 64 bits en hexadecimal (detecta copias reescaladas)
    grupo: str = ""             # fotos casi iguales comparten grupo
    incluir: bool = True
    motivo: str = ""            # por qué NO se incluye (vacío si se incluye)
    avisos: str = ""            # observaciones separadas por ";" (no excluyen la foto)
    split: str = ""             # "entrenamiento" o "prueba" (vacío si no se incluye)
    muestra_mascara: bool = False  # True si su máscara se marca a mano para evaluar la segmentación
    # Miniatura de color de 12 x 12 (432 bytes). Solo vive en memoria durante el escaneo: no se guarda en el CSV.
    firma: bytes = field(default=b"", repr=False, compare=False)


COLUMNAS = [f.name for f in fields(Registro) if f.name != "firma"]
ANCHO_FIRMA = 12


# ------------------------------------------------------------ huella visual
def hash_promedio(img: np.ndarray) -> int:
    """Huella visual de 64 bits ("average hash").

    Se reduce la foto a una cuadrícula de 8 x 8 en gris y cada celda vale 1 si es más clara que el
    promedio. Dos copias de la misma foto (reescalada, recomprimida, con un poco más de brillo) dan
    huellas casi iguales. No detecta recortes, giros ni fotos distintas de la misma prenda.
    """
    pequena = redimensionar_lado_max(img, 64)
    lum = luminancia(pequena)
    cuadricula = remuestrear_bilineal(lum[..., None], 8, 8)[..., 0].astype(np.float32)
    bits = (cuadricula > cuadricula.mean()).ravel()
    valor = 0
    for b in bits:
        valor = (valor << 1) | int(b)
    return valor


def distancia_hamming(a: int, b: int) -> int:
    """Número de bits distintos entre dos huellas (0 = idénticas, 64 = opuestas)."""
    return bin(a ^ b).count("1")


def firma_color(img: np.ndarray) -> bytes:
    """Miniatura de color de 12 x 12 (432 bytes) para comparar fotos por su color.

    El hash de 64 bits solo ve la silueta en gris: una camiseta negra y una azul del mismo corte sobre
    fondo blanco dan la misma huella. Esta miniatura conserva el color y distingue ambas.
    """
    pequena = redimensionar_lado_max(img, 64)
    return remuestrear_bilineal(pequena, ANCHO_FIRMA, ANCHO_FIRMA).tobytes()


def diferencia_firmas(a: bytes, b: bytes) -> float:
    """Diferencia media absoluta entre dos miniaturas (0-255). Copias de la misma foto: menos de ~2."""
    va, vb = np.frombuffer(a, np.uint8).astype(np.int16), np.frombuffer(b, np.uint8).astype(np.int16)
    return float(np.abs(va - vb).mean())


# ------------------------------------------------------------------ escaneo
def escanear(raiz: str | Path, calidad: CalidadConfig | None = None) -> list[Registro]:
    """Recorre la base y crea un `Registro` por imagen. Una foto ilegible queda excluida (no frena el lote)."""
    calidad = calidad or CalidadConfig()
    raiz = Path(raiz)
    if not raiz.is_dir():
        raise NotADirectoryError(f"No es una carpeta: {raiz}")

    registros = []
    for ruta in listar_imagenes(raiz):
        rel = ruta.relative_to(raiz).as_posix()
        partes = rel.split("/")
        reg = Registro(ruta=rel, categoria=partes[0] if len(partes) > 1 else SIN_CATEGORIA)
        try:
            reg.sha1 = hashlib.sha1(ruta.read_bytes()).hexdigest()
            img = cargar_imagen(ruta)
            reg.alto, reg.ancho = (int(v) for v in img.shape[:2])
            reg.ahash = f"{hash_promedio(img):016x}"
            reg.firma = firma_color(img)
        except Exception as e:  # archivo dañado, formato raro...
            reg.incluir, reg.motivo = False, f"error: {type(e).__name__}"
            registros.append(reg)
            continue
        if min(reg.alto, reg.ancho) < calidad.lado_min:
            _avisar(reg, "baja_resolucion")
        registros.append(reg)
    return registros


def _avisar(reg: Registro, aviso: str) -> None:
    avisos = [a for a in reg.avisos.split(";") if a]
    if aviso not in avisos:
        avisos.append(aviso)
    reg.avisos = ";".join(avisos)


# --------------------------------------------------------------- exclusiones
def aplicar_exclusiones(registros: list[Registro], rutas: set[str]) -> None:
    """Excluye las fotos cuya ruta está en `rutas` (por ejemplo, fotos con una persona puesta)."""
    for r in registros:
        if r.incluir and r.ruta in rutas:
            r.incluir, r.motivo = False, MOTIVO_MANUAL


def leer_exclusiones(ruta: str | Path) -> set[str]:
    """Lee un .txt con una ruta por línea (líneas vacías o que empiezan con # se ignoran)."""
    ruta = Path(ruta)
    if not ruta.exists():
        return set()
    lineas = ruta.read_text(encoding="utf-8-sig").splitlines()
    return {l.strip().replace("\\", "/") for l in lineas if l.strip() and not l.lstrip().startswith("#")}


# --------------------------------------------------------------- duplicados
def marcar_duplicados(registros: list[Registro], cfg: DatasetConfig | None = None) -> None:
    """Excluye copias exactas, agrupa casi duplicados y asigna `grupo` a todas las fotos incluidas."""
    cfg = cfg or DatasetConfig()

    # 1) copias exactas: se conserva la primera (por ruta)
    primera: dict[str, Registro] = {}
    for r in sorted((r for r in registros if r.incluir and r.sha1), key=lambda r: r.ruta):
        if r.sha1 not in primera:
            primera[r.sha1] = r
            continue
        original = primera[r.sha1]
        r.incluir, r.motivo = False, f"duplicado_exacto_de:{original.ruta}"
        if r.categoria != original.categoria:
            _avisar(original, "duplicado_en_otra_categoria")

    # 2) casi duplicados: componentes conexas de "huellas a distancia <= umbral" (unión-búsqueda)
    vigentes = sorted((r for r in registros if r.incluir and r.ahash), key=lambda r: r.ruta)
    huellas = [int(r.ahash, 16) for r in vigentes]
    padre = list(range(len(vigentes)))

    def raiz_de(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    def parecidas(i: int, j: int) -> bool:
        """Misma foto reescalada o recomprimida: silueta igual, mismo color y misma proporción."""
        if distancia_hamming(huellas[i], huellas[j]) > cfg.umbral_casi_duplicado:
            return False
        a, b = vigentes[i], vigentes[j]
        if a.alto and b.alto and abs(a.ancho / a.alto - b.ancho / b.alto) > 0.03 * (a.ancho / a.alto):
            return False                      # proporciones distintas: no es una copia reescalada
        if a.firma and b.firma and diferencia_firmas(a.firma, b.firma) > cfg.umbral_firma_color:
            return False                      # misma silueta pero otro color o estampado: prenda distinta
        return True

    for i in range(len(vigentes)):
        for j in range(i + 1, len(vigentes)):
            if parecidas(i, j):
                padre[raiz_de(j)] = raiz_de(i)

    componentes: dict[int, list[Registro]] = defaultdict(list)
    for i, r in enumerate(vigentes):
        componentes[raiz_de(i)].append(r)
    for miembros in componentes.values():
        grupo = "g_" + min(m.sha1 for m in miembros)[:8]
        for m in miembros:
            m.grupo = grupo
            if len(miembros) > 1:
                _avisar(m, "casi_duplicado")
                if len({x.categoria for x in miembros}) > 1:
                    _avisar(m, "casi_duplicado_otra_categoria")

    for r in registros:
        if not r.incluir:
            r.grupo = ""


# ---------------------------------------------------------------------- ids
def _prefijo(categoria: str) -> str:
    sin_tildes = unicodedata.normalize("NFD", categoria.lower())
    letras = "".join(c for c in sin_tildes if "a" <= c <= "z")
    return letras[:3] or "img"


def asignar_ids(registros: list[Registro], previos: list[Registro] | None = None) -> None:
    """Da a cada foto un id corto y estable (cam_001). Una foto ya conocida conserva el suyo."""
    previos = previos or []
    por_ruta = {p.ruta: p.id for p in previos if p.id}
    conteo_sha1 = Counter(p.sha1 for p in previos if p.sha1)
    por_sha1 = {p.sha1: p.id for p in previos if p.id and p.sha1 and conteo_sha1[p.sha1] == 1}

    siguiente: dict[str, int] = defaultdict(int)
    for p in previos:
        prefijo, _, numero = p.id.rpartition("_")
        if prefijo and numero.isdigit():
            siguiente[prefijo] = max(siguiente[prefijo], int(numero))

    usados: set[str] = set()
    pendientes = []
    for r in registros:
        previo = por_ruta.get(r.ruta) or por_sha1.get(r.sha1, "")
        if previo and previo not in usados:
            r.id = previo
            usados.add(previo)
        else:
            pendientes.append(r)
    for r in sorted(pendientes, key=lambda r: r.ruta):
        prefijo = _prefijo(r.categoria)
        siguiente[prefijo] += 1
        r.id = f"{prefijo}_{siguiente[prefijo]:03d}"
        usados.add(r.id)


# ----------------------------------------------------------------- división
def dividir(registros: list[Registro], cfg: DatasetConfig | None = None,
            previos: list[Registro] | None = None) -> None:
    """Reparte las fotos incluidas en entrenamiento y prueba, categoría por categoría.

    - Estratificada: cada categoría aporta aproximadamente `fraccion_prueba` a la prueba.
    - Por grupos: los casi duplicados nunca quedan separados.
    - Estable: lo que ya estaba asignado en `previos` no se mueve; las fotos nuevas se reparten
      para acercar cada categoría a la proporción pedida.
    - Una categoría con un solo grupo se queda toda en entrenamiento (no hay con qué evaluar).
    """
    cfg = cfg or DatasetConfig()
    previo_split = {p.ruta: p.split for p in (previos or []) if p.split}
    por_categoria: dict[str, dict[str, list[Registro]]] = defaultdict(lambda: defaultdict(list))
    for r in registros:
        r.split = ""
        if r.incluir and r.grupo:
            por_categoria[r.categoria][r.grupo].append(r)

    for grupos in por_categoria.values():
        total = sum(len(m) for m in grupos.values())
        objetivo = cfg.fraccion_prueba * total
        en_prueba = 0
        pendientes = []
        for grupo, miembros in grupos.items():
            fijo = next((previo_split[m.ruta] for m in miembros if m.ruta in previo_split), "")
            if fijo:
                for m in miembros:
                    m.split = fijo
                en_prueba += len(miembros) if fijo == PRUEBA else 0
            else:
                pendientes.append(grupo)

        # orden pseudoaleatorio pero reproducible (depende de la semilla y del grupo)
        pendientes.sort(key=lambda g: hashlib.sha1(f"{cfg.semilla}-{g}".encode()).hexdigest())
        for grupo in pendientes:
            tam = len(grupos[grupo])
            va_a_prueba = (len(grupos) >= 2
                           and abs(en_prueba + tam - objetivo) < abs(en_prueba - objetivo))
            destino = PRUEBA if va_a_prueba else ENTRENAMIENTO
            en_prueba += tam if va_a_prueba else 0
            for m in grupos[grupo]:
                m.split = destino


# ------------------------------------------------- muestra de máscaras a mano
def elegir_muestra(registros: list[Registro], cfg: DatasetConfig | None = None) -> None:
    """Marca (`muestra_mascara`) las fotos cuya máscara se dibuja a mano para evaluar la segmentación.

    Se reparte por igual entre categorías y por separado en entrenamiento (para ajustar parámetros)
    y prueba (para la medición final). Una sola foto por grupo de casi duplicados. Si ya había una muestra
    elegida, se respeta y solo se completa; el orden aleatorio no depende del tamaño pedido, así que pedir
    una muestra mayor conserva las máscaras que ya dibujaste.
    """
    cfg = cfg or DatasetConfig()
    for r in registros:
        if not (r.incluir and r.split):
            r.muestra_mascara = False

    for split, n in ((ENTRENAMIENTO, cfg.mascaras_entrenamiento), (PRUEBA, cfg.mascaras_prueba)):
        candidatas = [r for r in registros if r.incluir and r.split == split]
        categorias = sorted({r.categoria for r in candidatas})
        if not categorias:
            continue
        base, resto = divmod(n, len(categorias))
        for j, categoria in enumerate(categorias):
            cupo = base + (1 if j < resto else 0)
            de_categoria = sorted((r for r in candidatas if r.categoria == categoria), key=lambda r: r.ruta)
            barajadas = list(de_categoria)
            random.Random(f"{cfg.semilla}-{split}-{categoria}").shuffle(barajadas)
            orden = [r for r in de_categoria if r.muestra_mascara] + [r for r in barajadas if not r.muestra_mascara]
            for r in de_categoria:
                r.muestra_mascara = False
            grupos_usados: set[str] = set()
            elegidas = 0
            for r in orden:
                if elegidas >= cupo:
                    break
                if r.grupo in grupos_usados:
                    continue
                grupos_usados.add(r.grupo)
                r.muestra_mascara = True
                elegidas += 1


def nombre_en_muestra(reg: Registro) -> str:
    """Nombre con el que una foto de la muestra se copia: <categoria>__<archivo>.

    El anotador de máscaras guarda `<nombre>_gt.png` con este mismo nombre base.
    """
    return f"{reg.categoria}__{Path(reg.ruta).name}"


def copiar_muestra(registros: list[Registro], raiz: str | Path, destino: str | Path) -> tuple[int, int]:
    """Copia las fotos de la muestra a `destino` (los originales no se tocan). Devuelve (nuevas, total)."""
    raiz, destino = Path(raiz), Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    nuevas = total = 0
    for r in registros:
        if not (r.incluir and r.muestra_mascara):
            continue
        total += 1
        salida = destino / nombre_en_muestra(r)
        if not salida.exists():
            shutil.copy2(raiz / r.ruta, salida)
            nuevas += 1
    return nuevas, total


# ---------------------------------------------------------------------- CSV
def guardar_csv(registros: list[Registro], ruta: str | Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel abre bien los acentos
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        for r in registros:
            fila = {c: getattr(r, c) for c in COLUMNAS}
            fila["incluir"] = "si" if r.incluir else "no"
            fila["muestra_mascara"] = "si" if r.muestra_mascara else ""
            w.writerow(fila)


def _si(valor: str) -> bool:
    return valor.strip().lower() in {"si", "sí", "s", "true", "1", "yes", "y"}


def cargar_csv(ruta: str | Path) -> list[Registro]:
    """Lee un dataset.csv (admite columnas faltantes y "si/no" escritos a mano, por ejemplo desde Excel)."""
    registros = []
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            fila = {k: (v or "") for k, v in fila.items() if k}
            registros.append(Registro(
                id=fila.get("id", ""), ruta=fila.get("ruta", "").replace("\\", "/"),
                categoria=fila.get("categoria", ""),
                ancho=int(fila.get("ancho") or 0), alto=int(fila.get("alto") or 0),
                sha1=fila.get("sha1", ""), ahash=fila.get("ahash", ""), grupo=fila.get("grupo", ""),
                incluir=_si(fila.get("incluir", "si")) if fila.get("incluir", "") else True,
                motivo=fila.get("motivo", ""), avisos=fila.get("avisos", ""),
                split=fila.get("split", ""), muestra_mascara=_si(fila.get("muestra_mascara", "")),
            ))
    return registros


# ------------------------------------------------------------ orquestación
def construir_dataset(raiz: str | Path, cfg: DatasetConfig | None = None,
                      previos: list[Registro] | None = None,
                      exclusiones: set[str] | None = None,
                      calidad: CalidadConfig | None = None,
                      conservar_division: bool = True) -> list[Registro]:
    """Escanea la base y produce el inventario completo (ids, duplicados, división y muestra).

    `previos` es el dataset.csv anterior (si existe): conserva ids, división, muestra y las exclusiones
    hechas a mano (filas con `incluir = no` y sin otro motivo).

    `conservar_division=False` rehace la división y la muestra desde cero (los ids se conservan). Úsalo
    solo al principio, después de descartar fotos y antes de empezar a marcar máscaras: una vez que
    ajustas parámetros con el conjunto de entrenamiento, mover fotos entre conjuntos invalida la prueba.
    """
    cfg = cfg or DatasetConfig()
    previos = previos or []
    registros = escanear(raiz, calidad)

    manuales = {p.ruta for p in previos if not p.incluir and p.motivo in ("", MOTIVO_MANUAL)}
    aplicar_exclusiones(registros, manuales | (exclusiones or set()))
    marcar_duplicados(registros, cfg)
    asignar_ids(registros, previos)
    dividir(registros, cfg, previos if conservar_division else None)

    muestra_previa = {p.ruta for p in previos if p.muestra_mascara} if conservar_division else set()
    for r in registros:
        r.muestra_mascara = r.ruta in muestra_previa
    elegir_muestra(registros, cfg)
    return registros


def resumen(registros: list[Registro]) -> str:
    """Tabla de texto: por categoría, cuántas fotos hay y cómo quedaron repartidas."""
    cols = ["categoria", "total", "incluidas", "entrenam.", "prueba", "excluidas", "con_avisos", "mascaras"]
    filas = []
    for categoria in sorted({r.categoria for r in registros}) + ["TOTAL"]:
        rs = registros if categoria == "TOTAL" else [r for r in registros if r.categoria == categoria]
        inc = [r for r in rs if r.incluir]
        filas.append([
            categoria, len(rs), len(inc),
            sum(r.split == ENTRENAMIENTO for r in inc), sum(r.split == PRUEBA for r in inc),
            len(rs) - len(inc), sum(bool(r.avisos) for r in inc), sum(r.muestra_mascara for r in inc),
        ])
    anchos = [max(len(str(x)) for x in [c] + [f[i] for f in filas]) for i, c in enumerate(cols)]
    linea = lambda vals: "  ".join(str(v).ljust(anchos[0]) if i == 0 else str(v).rjust(anchos[i])
                                   for i, v in enumerate(vals))
    return "\n".join([linea(cols), "-" * (sum(anchos) + 2 * (len(cols) - 1))] + [linea(f) for f in filas])

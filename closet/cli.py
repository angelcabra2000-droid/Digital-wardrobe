"""Línea de comandos del proyecto.

Uso (desde la raíz del proyecto):
    python -m closet inventario "Aplicacion base de datos" --revision      (paso 0)
    python -m closet muestra "Aplicacion base de datos"                    (paso 0)
    python -m closet preprocesar "Aplicacion base de datos" --salida resultados/preprocesado
    python -m closet preprocesar "ruta/foto.jpg" --salida resultados/prueba --comparar
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .config import DatasetConfig, PreprocesoConfig
from .dataset import (ENTRENAMIENTO, PRUEBA, cargar_csv, construir_dataset, copiar_muestra,
                      elegir_muestra, guardar_csv, leer_exclusiones, resumen)
from .imagen_io import cargar_imagen, guardar_imagen, listar_imagenes
from .preprocesamiento import preprocesar_imagen
from .revision import generar_revision

COLUMNAS_CSV = [
    "ruta", "categoria", "ancho_original", "alto_original", "ancho", "alto",
    "nitidez", "brillo", "contraste", "frac_saturada", "frac_oscura",
    "ganancia_r", "ganancia_g", "ganancia_b", "advertencias",
]


def _categoria(imagen: Path, raiz: Path) -> str:
    """La categoría es la primera subcarpeta bajo la raíz (camisas, chaquetas, pantalones...)."""
    if raiz.is_dir():
        partes = imagen.relative_to(raiz).parts
        if len(partes) > 1:
            return partes[0]
    return imagen.parent.name


def _ruta_relativa(imagen: Path, raiz: Path) -> Path:
    return imagen.relative_to(raiz) if raiz.is_dir() else Path(imagen.name)


def comando_preprocesar(args: argparse.Namespace) -> int:
    raiz = Path(args.ruta)
    if not raiz.exists():
        print(f"No existe la ruta: {raiz}", file=sys.stderr)
        return 1
    imagenes = listar_imagenes(raiz)
    if not imagenes:
        print(f"No se encontraron imágenes en: {raiz}", file=sys.stderr)
        return 1

    cfg = PreprocesoConfig(
        lado_max=args.lado_max,
        balance_blancos=not args.sin_balance,
        estirar_contraste=not args.sin_contraste,
        sigma_suavizado=args.sigma,
    )
    salida = Path(args.salida)
    filas, conteo = [], Counter()

    for i, ruta in enumerate(imagenes, start=1):
        rel = _ruta_relativa(ruta, raiz)
        fila = {c: "" for c in COLUMNAS_CSV}
        fila["ruta"], fila["categoria"] = str(rel), _categoria(ruta, raiz)
        try:
            img = cargar_imagen(ruta)
            fila["alto_original"], fila["ancho_original"] = img.shape[:2]
            res = preprocesar_imagen(img, cfg)
        except Exception as e:  # un archivo dañado no debe frenar el lote
            fila["advertencias"] = f"error: {e}"
            conteo["error"] += 1
            print(f"[{i}/{len(imagenes)}] {rel}  ERROR: {e}", file=sys.stderr)
            filas.append(fila)
            continue

        destino = salida / rel.with_name(rel.stem + "_norm.png")
        guardar_imagen(res.normalizada, destino)
        if args.comparar:
            lado_a_lado = np.hstack([res.original, res.normalizada])
            guardar_imagen(lado_a_lado, salida / rel.with_name(rel.stem + "_comparacion.png"))

        c = res.calidad
        ganancias = res.pasos.get("ganancias_rgb", ("", "", ""))
        fila.update(
            ancho=c["ancho"], alto=c["alto"],
            nitidez=round(c["nitidez"], 1), brillo=round(c["brillo"], 1),
            contraste=round(c["contraste"], 1),
            frac_saturada=round(c["frac_saturada"], 3), frac_oscura=round(c["frac_oscura"], 3),
            ganancia_r=ganancias[0], ganancia_g=ganancias[1], ganancia_b=ganancias[2],
            advertencias=";".join(c["advertencias"]),
        )
        conteo.update(c["advertencias"])
        filas.append(fila)
        aviso = f"  [{', '.join(c['advertencias'])}]" if c["advertencias"] else ""
        print(f"[{i}/{len(imagenes)}] {rel}{aviso}")

    salida.mkdir(parents=True, exist_ok=True)
    csv_ruta = salida / "calidad.csv"
    with open(csv_ruta, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel abre bien los acentos
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"\n{len(imagenes)} imágenes procesadas -> {salida}")
    print(f"Reporte de calidad: {csv_ruta}")
    if conteo:
        print("Advertencias: " + ", ".join(f"{k}={v}" for k, v in sorted(conteo.items())))
    return 0


def comando_inventario(args: argparse.Namespace) -> int:
    raiz = Path(args.ruta)
    if not raiz.is_dir():
        print(f"No existe la carpeta: {raiz}", file=sys.stderr)
        return 1
    if not 0 < args.fraccion_prueba < 1:
        print("--fraccion-prueba debe estar entre 0 y 1 (por ejemplo 0.3).", file=sys.stderr)
        return 1

    datos = Path(args.datos)
    csv_ruta = datos / "dataset.csv"
    previos = cargar_csv(csv_ruta) if csv_ruta.exists() else []
    ruta_excl = Path(args.exclusiones) if args.exclusiones else datos / "exclusiones.txt"
    exclusiones = leer_exclusiones(ruta_excl)

    cfg = DatasetConfig(fraccion_prueba=args.fraccion_prueba, semilla=args.semilla)
    print(f"Escaneando {raiz} ...")
    registros = construir_dataset(raiz, cfg, previos, exclusiones, conservar_division=not args.redividir)
    guardar_csv(registros, csv_ruta)

    print("\n" + resumen(registros))
    categorias = sorted({r.categoria for r in registros})
    if len(categorias) == 1:
        print(f"\nOJO: solo se detectó una categoría ({categorias[0]!r}). La carpeta que le pasas al comando debe "
              "contener directamente\nlas subcarpetas camisas, chaquetas, pantalones... "
              + ("Parece que hay una carpeta anidada con el mismo nombre." if categorias[0] == raiz.name else
                 "Revisa la ruta."))
    motivos = Counter(r.motivo.split(":")[0] for r in registros if not r.incluir)
    avisos = Counter(a for r in registros if r.incluir for a in r.avisos.split(";") if a)
    if motivos:
        print("\nExcluidas: " + ", ".join(f"{k}={v}" for k, v in sorted(motivos.items())))
    if avisos:
        print("Avisos:    " + ", ".join(f"{k}={v}" for k, v in sorted(avisos.items())))
    if previos and args.redividir:
        print("\n(se conservaron los ids; la división y la muestra se rehicieron desde cero)")
    elif previos:
        print(f"\n(se conservaron los ids, la división y la muestra del {csv_ruta.name} anterior)")
    print(f"\nInventario: {csv_ruta}")

    if args.revision:
        html = generar_revision(registros, raiz, datos / "revision")
        print(f"Hoja de revisión: {html}")
        print("  Ábrela con doble clic, marca las fotos que no sirven y descarga exclusiones.txt;")
        print(f"  guárdalo en {datos}/ y vuelve a correr este mismo comando con --redividir")
        print("  (solo ahora, antes de marcar máscaras: rehace la división con las fotos que quedan).")
    return 0


def comando_muestra(args: argparse.Namespace) -> int:
    raiz, csv_ruta = Path(args.ruta), Path(args.datos) / "dataset.csv"
    if not raiz.is_dir():
        print(f"No existe la carpeta: {raiz}", file=sys.stderr)
        return 1
    if not csv_ruta.exists():
        print(f"Falta {csv_ruta}. Corre primero: python -m closet inventario \"{raiz}\"", file=sys.stderr)
        return 1

    registros = cargar_csv(csv_ruta)
    cfg = DatasetConfig(mascaras_entrenamiento=args.n_entrenamiento, mascaras_prueba=args.n_prueba,
                        semilla=args.semilla)
    elegir_muestra(registros, cfg)
    guardar_csv(registros, csv_ruta)
    nuevas, total = copiar_muestra(registros, raiz, args.salida)

    print(f"Muestra para marcar máscaras a mano: {total} fotos en {args.salida}  ({nuevas} copiadas nuevas)")
    for split in (ENTRENAMIENTO, PRUEBA):
        por_cat = Counter(r.categoria for r in registros if r.muestra_mascara and r.split == split)
        print(f"  {split + ':':<15}" + ", ".join(f"{c}={n}" for c, n in sorted(por_cat.items())))
    print(f"\nSiguiente: abre herramientas/anotador_mascaras.html en Chrome o Edge, elige todas las fotos de "
          f"{args.salida}\ny dibuja el contorno de cada prenda. Guarda las máscaras (_gt.png) en esa misma carpeta.")
    return 0


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="closet", description="Closet Digital: segmentación y clasificación de prendas")
    sub = p.add_subparsers(dest="comando", required=True)

    inv = sub.add_parser("inventario", help="paso 0: inventaría la base y la divide en entrenamiento / prueba")
    inv.add_argument("ruta", help="carpeta de la base (con una subcarpeta por categoría)")
    inv.add_argument("--datos", default="datos", help="carpeta donde se guarda dataset.csv (por defecto %(default)s)")
    inv.add_argument("--fraccion-prueba", type=float, default=DatasetConfig.fraccion_prueba,
                     help="parte de la base reservada para la prueba final (por defecto %(default)s)")
    inv.add_argument("--semilla", default=DatasetConfig.semilla,
                     help="texto que fija la división; no lo cambies una vez empezado el proyecto")
    inv.add_argument("--exclusiones", help="archivo .txt con rutas a excluir (por defecto datos/exclusiones.txt)")
    inv.add_argument("--redividir", action="store_true",
                     help="rehacer la división y la muestra desde cero (solo antes de empezar a marcar máscaras)")
    inv.add_argument("--revision", action="store_true",
                     help="generar la hoja de revisión visual (miniaturas) para descartar fotos")
    inv.set_defaults(funcion=comando_inventario)

    mue = sub.add_parser("muestra", help="paso 0: elige y copia las fotos cuya máscara se marca a mano")
    mue.add_argument("ruta", help="carpeta de la base (la misma del inventario)")
    mue.add_argument("--datos", default="datos", help="carpeta con dataset.csv (por defecto %(default)s)")
    mue.add_argument("--salida", default="muestra", help="carpeta donde copiar la muestra (por defecto %(default)s)")
    mue.add_argument("--n-entrenamiento", type=int, default=DatasetConfig.mascaras_entrenamiento,
                     help="fotos de entrenamiento a marcar (por defecto %(default)s)")
    mue.add_argument("--n-prueba", type=int, default=DatasetConfig.mascaras_prueba,
                     help="fotos de prueba a marcar (por defecto %(default)s)")
    mue.add_argument("--semilla", default=DatasetConfig.semilla, help="debe ser la misma del inventario")
    mue.set_defaults(funcion=comando_muestra)

    pre = sub.add_parser("preprocesar", help="unifica las fotos y genera un reporte de calidad")
    pre.add_argument("ruta", help="una imagen o una carpeta (se recorren las subcarpetas)")
    pre.add_argument("--salida", default="resultados/preprocesado", help="carpeta de resultados")
    pre.add_argument("--lado-max", type=int, default=PreprocesoConfig.lado_max,
                     help="lado mayor de trabajo en px (por defecto %(default)s)")
    pre.add_argument("--sigma", type=float, default=PreprocesoConfig.sigma_suavizado,
                     help="sigma del suavizado gaussiano; 0 lo desactiva (por defecto %(default)s)")
    pre.add_argument("--sin-balance", action="store_true", help="no corregir el balance de blancos")
    pre.add_argument("--sin-contraste", action="store_true", help="no estirar el contraste")
    pre.add_argument("--comparar", action="store_true",
                     help="guardar también una imagen original | normalizada lado a lado")
    pre.set_defaults(funcion=comando_preprocesar)
    return p


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    return args.funcion(args)


if __name__ == "__main__":
    raise SystemExit(main())

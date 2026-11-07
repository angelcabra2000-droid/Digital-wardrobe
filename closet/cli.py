"""Línea de comandos del proyecto.

Uso (desde la raíz del proyecto):
    python -m closet inventario "Aplicacion base de datos" --revision      (paso 0)
    python -m closet muestra "Aplicacion base de datos"                    (paso 0)
    python -m closet preprocesar "Aplicacion base de datos" --salida resultados/preprocesado
    python -m closet preprocesar "ruta/foto.jpg" --salida resultados/prueba --comparar
    python -m closet segmentar "ruta/foto.jpg" --salida resultados/seg                  (paso 2)
    python -m closet evaluar --muestra muestra                                         (paso 2: IoU, entrenamiento)
    python -m closet normalizar "Aplicacion base de datos" --salida resultados/norm    (paso 3)
    python -m closet caracteristicas "Aplicacion base de datos"                        (paso 4)
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .caracteristicas import extraer_caracteristicas
from .config import DatasetConfig, NormalizacionConfig, PreprocesoConfig, SegmentacionConfig
from .dataset import (ENTRENAMIENTO, PRUEBA, cargar_csv, construir_dataset, copiar_muestra,
                      elegir_muestra, guardar_csv, leer_exclusiones, resumen)
from .imagen_io import cargar_imagen, guardar_imagen, listar_imagenes
from .preprocesamiento import preprocesar_imagen
from .evaluacion import buscar_pares, evaluar, resumen as resumen_evaluacion, superposicion
from .pipeline import procesar_foto
from .revision import generar_revision
from .segmentacion import imagen_clusters, recortar_fondo, segmentar_archivo

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


def _config_segmentacion(args: argparse.Namespace) -> SegmentacionConfig:
    d = SegmentacionConfig()
    return SegmentacionConfig(k=args.k, semilla=d.semilla, peso_textura=args.textura, peso_espacial=args.espacial,
                              suavizado=args.suavizado, nucleo=args.nucleo, cierre=args.cierre,
                              ventana=args.ventana, umbral_centro=args.umbral_centro,
                              refinar_otsu=args.otsu, banda_otsu=args.banda_otsu)


def _solo_incluidas(imagenes: list[Path], raiz: Path, datos: str, todas: bool) -> list[Path]:
    """Quita las fotos que excluiste en dataset.csv (incluir = no), salvo que se pida --todas."""
    csv_ruta = Path(datos) / "dataset.csv"
    if todas or not raiz.is_dir() or not csv_ruta.exists():
        return imagenes
    excluidas = {r.ruta.replace("\\", "/") for r in cargar_csv(csv_ruta) if not r.incluir}
    quedan = [p for p in imagenes if _ruta_relativa(p, raiz).as_posix() not in excluidas]
    if len(quedan) < len(imagenes):
        print(f"Se omiten {len(imagenes) - len(quedan)} fotos excluidas en {csv_ruta} (usa --todas para procesarlas también)\n")
    return quedan


def comando_segmentar(args: argparse.Namespace) -> int:
    raiz = Path(args.ruta)
    if not raiz.exists():
        print(f"No existe la ruta: {raiz}", file=sys.stderr)
        return 1
    imagenes = _solo_incluidas(listar_imagenes(raiz), raiz, args.datos, args.todas)
    salida = Path(args.salida)
    cfg_pre, cfg = PreprocesoConfig(lado_max=args.lado_max), _config_segmentacion(args)
    sin_prenda = 0
    for ruta in imagenes:
        pre, res = segmentar_archivo(ruta, cfg_pre, cfg, preprocesar=not args.sin_preproceso)
        destino = salida / _ruta_relativa(ruta, raiz).parent
        guardar_imagen(imagen_clusters(res, pre.original), destino / f"{ruta.stem}_clusters.png")
        if not res.encontrada:
            sin_prenda += 1
            print(f"  {ruta.name}: NO se encontró prenda (revisa {ruta.stem}_clusters.png)")
            continue
        guardar_imagen((res.mascara * 255).astype(np.uint8), destino / f"{ruta.stem}_mascara.png")
        guardar_imagen(recortar_fondo(pre.original, res.mascara), destino / f"{ruta.stem}_sin_fondo.png")
        print(f"  {ruta.name}: prenda = {100 * res.mascara.mean():.1f}% de la imagen")
    print(f"\n{len(imagenes) - sin_prenda}/{len(imagenes)} segmentadas. Resultados en {salida}")
    return 0 if sin_prenda == 0 else 2


def comando_evaluar(args: argparse.Namespace) -> int:
    csv_ruta = Path(args.datos) / "dataset.csv"
    if not csv_ruta.exists() or not Path(args.muestra).is_dir():
        print(f"Faltan {csv_ruta} o la carpeta {args.muestra}", file=sys.stderr)
        return 1
    if args.conjunto == PRUEBA and not args.prueba_final:
        print("Las máscaras de PRUEBA se miden una sola vez, al final. Si de verdad es el momento, "
              "agrega --prueba-final.", file=sys.stderr)
        return 1
    pares, faltan = buscar_pares(cargar_csv(csv_ruta), args.muestra, args.conjunto)
    for f in faltan:
        print(f"  falta foto o máscara: {f}", file=sys.stderr)
    if not pares:
        print("No hay pares foto/máscara para evaluar.", file=sys.stderr)
        return 1

    cfg_pre, cfg = PreprocesoConfig(lado_max=args.lado_max), _config_segmentacion(args)
    salida = Path(args.salida)
    print(f"Evaluando {len(pares)} fotos del conjunto '{args.conjunto}' "
          f"({'sin' if args.sin_preproceso else 'con'} preprocesamiento)\n")
    print(f"{'foto':44} {'IoU':>6} {'prec':>6} {'recall':>6} {'área%':>6} {'real%':>6}")

    def reportar(m, res, rgb, gt):
        marca = "  <-- revisar" if m.iou < 0.8 else ""
        print(f"{m.par.foto.name:44} {m.iou:6.3f} {m.precision:6.3f} {m.recall:6.3f} "
              f"{100 * m.area_pred:6.1f} {100 * m.area_gt:6.1f}{marca}")
        guardar_imagen(superposicion(rgb, res.mascara, gt), salida / f"{m.par.foto.stem}_comparacion.png")

    mediciones = evaluar(pares, cfg_pre, cfg, not args.sin_preproceso, reportar)
    print(f"\n{'categoría':14} {'n':>3} {'IoU medio':>10} {'IoU mín':>8} {'prec':>6} {'recall':>6} {'F1':>6} {'<0.8':>5}")
    for cat, r in resumen_evaluacion(mediciones).items():
        print(f"{cat:14} {r['n']:3d} {r['iou']:10.3f} {r['iou_min']:8.3f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['f1']:6.3f} {r['bajo_080']:5d}")
    with open(salida / "evaluacion.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "foto", "categoria", "iou", "precision", "recall", "f1", "area_pred", "area_gt"])
        for m in mediciones:
            w.writerow([m.par.registro.id, m.par.foto.name, m.par.registro.categoria, f"{m.iou:.4f}",
                        f"{m.precision:.4f}", f"{m.recall:.4f}", f"{m.f1:.4f}",
                        f"{m.area_pred:.4f}", f"{m.area_gt:.4f}"])
    print(f"\nImágenes de comparación (verde acierto, rojo faltó, azul sobró) y evaluacion.csv en {salida}")
    return 0


def comando_normalizar(args: argparse.Namespace) -> int:
    raiz = Path(args.ruta)
    if not raiz.exists():
        print(f"No existe la ruta: {raiz}", file=sys.stderr)
        return 1
    imagenes = listar_imagenes(raiz)
    if not imagenes:
        print(f"No se encontraron imágenes en: {raiz}", file=sys.stderr)
        return 1
    imagenes = _solo_incluidas(imagenes, raiz, args.datos, args.todas)
    salida = Path(args.salida)
    cfg_norm = NormalizacionConfig(lado=args.lado, enderezar=not args.sin_enderezar)
    filas, fichas = [], []
    for ruta in imagenes:
        r = procesar_foto(ruta, cfg_norm=cfg_norm)
        cat = _categoria(ruta, raiz)
        filas.append([str(_ruta_relativa(ruta, raiz)), cat, " ".join(r.alertas), r.metricas.get("area", ""),
                      r.metricas.get("contraste", ""), r.metricas.get("fondo_sigma", ""),
                      r.metricas.get("giro_grados", ""), r.metricas.get("elongacion", ""),
                      r.metricas.get("ratio_aspecto", "")])
        if r.normalizada is None:
            print(f"  {ruta.name}: NO se encontró prenda")
            continue
        n = r.normalizada
        rgba = np.dstack([n.rgb, (n.mascara * 255).astype(np.uint8)])
        guardar_imagen(rgba, salida / _ruta_relativa(ruta, raiz).parent / f"{ruta.stem}_norm.png")
        fichas.append((n.rgb, bool(r.alertas)))
        aviso = f"  ALERTA: {' '.join(r.alertas)}" if r.alertas else ""
        print(f"  {ruta.name}: giro {n.meta['giro_grados']:>6}°  alto/ancho {n.meta['ratio_aspecto']:.2f}{aviso}")
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / "normalizacion.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ruta", "categoria", "alertas", "area", "contraste", "fondo_sigma", "giro_grados",
                    "elongacion", "ratio_aspecto"])
        w.writerows(filas)
    if fichas:
        guardar_imagen(_hoja_contacto(fichas, args.lado), salida / "hoja.png")
    con_alerta = sum(1 for f in filas if f[2])
    print(f"\n{len(fichas)}/{len(imagenes)} prendas normalizadas, {con_alerta} con alerta. "
          f"Resultados, normalizacion.csv y hoja.png en {salida}")
    return 0


def comando_caracteristicas(args: argparse.Namespace) -> int:
    raiz = Path(args.ruta)
    if not raiz.exists():
        print(f"No existe la ruta: {raiz}", file=sys.stderr)
        return 1
    imagenes = _solo_incluidas(listar_imagenes(raiz), raiz, args.datos, args.todas)
    if not imagenes:
        print(f"No se encontraron imágenes en: {raiz}", file=sys.stderr)
        return 1
    csv_datos = Path(args.datos) / "dataset.csv"
    conjunto = ({r.ruta.replace("\\", "/"): r.split for r in cargar_csv(csv_datos)}
                if csv_datos.exists() and raiz.is_dir() else {})
    filas, sin_prenda = [], []
    for ruta in imagenes:
        rel = _ruta_relativa(ruta, raiz).as_posix()
        r = procesar_foto(ruta)
        if r.normalizada is None:
            sin_prenda.append(rel)
            print(f"  {ruta.name}: NO se encontró prenda")
            continue
        c = extraer_caracteristicas(r.normalizada).como_dict()
        filas.append({"ruta": rel, "categoria": _categoria(ruta, raiz), "conjunto": conjunto.get(rel, ""),
                      "alertas": " ".join(r.alertas), **c})
        print(f"  {ruta.name}: {c['color_nombre']:<12} alto/ancho {c['ratio_aspecto']:.2f}  {c['estampado']}"
              + (f"  ALERTA: {' '.join(r.alertas)}" if r.alertas else ""))
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    if filas:
        with open(salida / "caracteristicas.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
    print(f"\n{len(filas)}/{len(imagenes)} prendas con características"
          + (f" ({len(sin_prenda)} sin prenda)" if sin_prenda else "")
          + f". caracteristicas.csv en {salida}")
    return 0


def _hoja_contacto(fichas, lado: int, columnas: int = 10, miniatura: int = 128) -> np.ndarray:
    """Todas las prendas normalizadas en una sola imagen (borde rojo = con alerta) para revisarlas a ojo."""
    paso = miniatura
    filas = (len(fichas) + columnas - 1) // columnas
    hoja = np.full((filas * paso, columnas * paso, 3), 60, dtype=np.uint8)
    for i, (rgb, alerta) in enumerate(fichas):
        f = max(1, lado // paso)
        t = rgb[::f, ::f][:paso, :paso]
        y, x = (i // columnas) * paso, (i % columnas) * paso
        hoja[y:y + t.shape[0], x:x + t.shape[1]] = np.where(t.any(axis=2, keepdims=True), t, 60)
        if alerta:
            hoja[y:y + 3, x:x + paso] = hoja[y + paso - 3:y + paso, x:x + paso] = (230, 40, 40)
            hoja[y:y + paso, x:x + 3] = hoja[y:y + paso, x + paso - 3:x + paso] = (230, 40, 40)
    return hoja


def _agregar_opciones_segmentacion(p: argparse.ArgumentParser) -> None:
    d = SegmentacionConfig()
    p.add_argument("--lado-max", type=int, default=PreprocesoConfig.lado_max, help="lado mayor de trabajo en px")
    p.add_argument("--sin-preproceso", action="store_true",
                   help="segmentar la foto solo redimensionada (sin balance de blancos, suavizado ni contraste)")
    p.add_argument("--k", type=int, default=d.k, help="clusters de K-means (por defecto %(default)s)")
    p.add_argument("--textura", type=float, default=d.peso_textura, help="peso de la textura local; 0 = solo color")
    p.add_argument("--espacial", type=float, default=d.peso_espacial, help="peso de la posición (x, y)")
    p.add_argument("--suavizado", type=float, default=d.suavizado, help="radio del filtro de moda (fracción del lado)")
    p.add_argument("--nucleo", type=float, default=d.nucleo, help="grosor mínimo de regiones (fracción del lado)")
    p.add_argument("--cierre", type=float, default=d.cierre, help="radio del cierre morfológico (fracción del lado)")
    p.add_argument("--ventana", type=float, default=d.ventana, help="ventana central donde se busca la prenda")
    p.add_argument("--umbral-centro", type=float, default=d.umbral_centro, help="lift mínimo en el centro")
    p.add_argument("--otsu", action="store_true", help="paso 2b: refinar el borde de la máscara con Otsu")
    p.add_argument("--banda-otsu", type=float, default=d.banda_otsu,
                   help="grosor de la banda que refina Otsu (fracción del lado mayor)")


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

    seg = sub.add_parser("segmentar", help="paso 2: separa la prenda del fondo (máscara y recorte sin fondo)")
    seg.add_argument("ruta", help="una imagen o una carpeta")
    seg.add_argument("--salida", default="resultados/segmentacion", help="carpeta de resultados")
    seg.add_argument("--datos", default="datos", help="carpeta con dataset.csv, de donde se leen las exclusiones (por defecto %(default)s)")
    seg.add_argument("--todas", action="store_true", help="procesar también las fotos excluidas en dataset.csv")
    _agregar_opciones_segmentacion(seg)
    seg.set_defaults(funcion=comando_segmentar)

    ev = sub.add_parser("evaluar", help="paso 2: mide la segmentación (IoU) contra las máscaras hechas a mano")
    ev.add_argument("--datos", default="datos", help="carpeta con dataset.csv (por defecto %(default)s)")
    ev.add_argument("--muestra", default="muestra", help="carpeta con las fotos y sus _gt.png (por defecto %(default)s)")
    ev.add_argument("--conjunto", default=ENTRENAMIENTO, choices=[ENTRENAMIENTO, PRUEBA],
                    help="qué máscaras usar (por defecto %(default)s)")
    ev.add_argument("--prueba-final", action="store_true", help="confirmar que ya es la medición final con prueba")
    ev.add_argument("--salida", default="resultados/evaluacion", help="carpeta de resultados")
    _agregar_opciones_segmentacion(ev)
    ev.set_defaults(funcion=comando_evaluar)

    nor = sub.add_parser("normalizar", help="paso 3: recorta, endereza y escala cada prenda (con alertas de segmentación dudosa)")
    nor.add_argument("ruta", help="una imagen o una carpeta")
    nor.add_argument("--salida", default="resultados/normalizacion", help="carpeta de resultados")
    nor.add_argument("--lado", type=int, default=NormalizacionConfig.lado, help="lado del lienzo final (por defecto %(default)s)")
    nor.add_argument("--datos", default="datos", help="carpeta con dataset.csv, de donde se leen las exclusiones (por defecto %(default)s)")
    nor.add_argument("--todas", action="store_true", help="procesar también las fotos excluidas en dataset.csv")
    nor.add_argument("--sin-enderezar", action="store_true", help="no girar la prenda, solo recortar y escalar")
    nor.set_defaults(funcion=comando_normalizar)

    car = sub.add_parser("caracteristicas", help="paso 4: color, forma y textura (Haralick) de cada prenda")
    car.add_argument("ruta", help="una imagen o una carpeta")
    car.add_argument("--salida", default="resultados/caracteristicas", help="carpeta de resultados (por defecto %(default)s)")
    car.add_argument("--datos", default="datos", help="carpeta con dataset.csv (por defecto %(default)s)")
    car.add_argument("--todas", action="store_true", help="procesar también las fotos excluidas en dataset.csv")
    car.set_defaults(funcion=comando_caracteristicas)
    return p


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    return args.funcion(args)


if __name__ == "__main__":
    raise SystemExit(main())

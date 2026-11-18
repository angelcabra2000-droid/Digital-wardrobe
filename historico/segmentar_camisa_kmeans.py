"""
Segmentación de prendas con K-means implementado desde cero.

Solo se usa NumPy (arreglos) y Pillow (leer/guardar la imagen).
Todo lo demás está escrito a mano:
  - conversión RGB -> Lab
  - textura local (desviación estándar en ventana, con imagen integral)
  - K-means (inicialización k-means++, asignación, actualización, convergencia)
  - filtro de moda para limpiar las etiquetas de K-means
  - selección de la prenda: clusters que dominan el centro de la foto + regiones pegadas
    a ella que no llegan al borde de la foto
  - morfología (dilatación / cierre) y relleno de huecos (inundación)

Uso:
    python segmentar_camisa_kmeans.py foto.jpg
    python segmentar_camisa_kmeans.py foto.jpg --k 4 --textura 3 --cierre 0.03 --suavizado 0.012
"""
import argparse
import os

import numpy as np
from PIL import Image


# ---------------------------------------------------------------- color
def rgb_a_lab(rgb):
    """rgb: (N,3) en [0,255] -> Lab (N,3). Iluminante D65."""
    c = rgb / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


# -------------------------------------------------------------- textura
def filtro_caja(a, r):
    """Promedio en ventana (2r+1)x(2r+1) usando imagen integral."""
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * r + 1
    s = c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]
    return s / (n * n)


def textura_local(L, r=3):
    """Desviación estándar local de la luminosidad: tela lisa ~0, superficies con relieve > 0."""
    m = filtro_caja(L, r)
    m2 = filtro_caja(L * L, r)
    return np.sqrt(np.maximum(m2 - m * m, 0))


# -------------------------------------------------------------- k-means
def kmeans(X, k, max_iter=100, tol=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # Inicialización k-means++
    centros = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centros)[None]) ** 2).sum(2), axis=1)
        p = d2 / d2.sum()
        centros.append(X[rng.choice(n, p=p)])
    C = np.array(centros, dtype=np.float64)

    for it in range(max_iter):
        # Asignación: distancia euclidiana al cuadrado a cada centroide
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None]
        etiquetas = np.argmin(d2, axis=1)

        # Actualización: media de cada cluster
        C_nuevo = C.copy()
        for j in range(k):
            miembros = X[etiquetas == j]
            if len(miembros) == 0:  # cluster vacío: reubicar en el punto más lejano
                C_nuevo[j] = X[np.argmax(d2.min(axis=1))]
            else:
                C_nuevo[j] = miembros.mean(axis=0)

        desplazamiento = np.abs(C_nuevo - C).max()
        C = C_nuevo
        if desplazamiento < tol:
            break

    return etiquetas, C, it + 1


# ------------------------------------------------------------ morfología
def _desplazar(m, dy, dx, relleno):
    out = np.full_like(m, relleno)
    h, w = m.shape
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[ys, xs] = m[yd, xd]
    return out


def erosion(m, it=1):
    for _ in range(it):
        r = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                r &= _desplazar(m, dy, dx, False)
        m = r
    return m


def dilatacion(m, it=1):
    for _ in range(it):
        r = m.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                r |= _desplazar(m, dy, dx, False)
        m = r
    return m


def dilatar_caja(m, r):
    """Dilatación con ventana cuadrada de lado 2r+1 (horizontal y luego vertical)."""
    out = m.copy()
    for d in range(1, r + 1):
        out |= _desplazar(m, 0, d, False) | _desplazar(m, 0, -d, False)
    tmp = out.copy()
    for d in range(1, r + 1):
        out |= _desplazar(tmp, d, 0, False) | _desplazar(tmp, -d, 0, False)
    return out


def dilatar_cruz(m, it):
    """Dilatación con vecindad en cruz (4 vecinos), repetida `it` veces = rombo."""
    for _ in range(it):
        r = m.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r |= _desplazar(m, dy, dx, False)
        m = r
    return m


def dilatar_octogono(m, r):
    """Cuadrado de radio ~0.4r seguido de rombo de radio ~0.6r: aproxima un círculo."""
    a = int(round(0.4 * r))
    return dilatar_cruz(dilatar_caja(m, a) if a > 0 else m, r - a)


def cerrar_caja(m, r):
    """Cierre: une partes separadas por huecos de hasta ~2r px (rayas, estampados, costuras)."""
    if r < 1:
        return m
    d = dilatar_octogono(m, r)
    return ~dilatar_octogono(~d, r)  # erosión = dilatar el complemento


# ----------------------------------------------- inundación / componentes
def inundar_desde_borde(m):
    """Píxeles True de m conectados (4 vecinos) con el borde de la imagen."""
    h, w = m.shape
    W = w + 2
    pad = np.zeros((h + 2, W), dtype=np.uint8)
    pad[1:-1, 1:-1] = m
    libre = bytearray(pad.tobytes())
    visto = bytearray(len(libre))
    borde = np.zeros((h + 2, W), dtype=bool)
    borde[1, 1:-1] = borde[-2, 1:-1] = True
    borde[1:-1, 1] = borde[1:-1, -2] = True
    pila = np.flatnonzero(borde.ravel() & pad.ravel().astype(bool)).tolist()
    for s in pila:
        visto[s] = 1
    while pila:
        i = pila.pop()
        for j in (i - 1, i + 1, i - W, i + W):
            if libre[j] and not visto[j]:
                visto[j] = 1
                pila.append(j)
    res = np.frombuffer(visto, dtype=np.uint8).reshape(h + 2, W)[1:-1, 1:-1]
    return res.astype(bool)


def rellenar_huecos(m):
    """Lo que no es alcanzable desde el borde por el exterior es un hueco interno."""
    return ~inundar_desde_borde(~m)


def componente_mas_grande(m):
    h, w = m.shape
    W = w + 2
    pad = np.zeros((h + 2, W), dtype=np.uint8)
    pad[1:-1, 1:-1] = m
    libre = bytearray(pad.tobytes())
    visto = bytearray(len(libre))
    mejor = []
    for s in np.flatnonzero(pad.ravel()).tolist():
        if visto[s]:
            continue
        visto[s] = 1
        comp, q = [s], 0
        while q < len(comp):
            i = comp[q]
            q += 1
            for j in (i - 1, i + 1, i - W, i + W):
                if libre[j] and not visto[j]:
                    visto[j] = 1
                    comp.append(j)
        if len(comp) > len(mejor):
            mejor = comp
    out = np.zeros(pad.size, dtype=bool)
    out[mejor] = True
    return out.reshape(h + 2, W)[1:-1, 1:-1]


# ------------------------------------------------------------- pipeline
def componentes_por_etiqueta(et):
    """Componentes 4-conexas de píxeles con la misma etiqueta (255 = barrera, se ignora).
    Devuelve (id de componente por píxel, -1 en barreras ; número de componentes)."""
    h, w = et.shape
    W = w + 2
    pad = np.full((h + 2, W), 255, dtype=np.uint8)
    pad[1:-1, 1:-1] = et
    lab = bytearray(pad.tobytes())
    comp = [-1] * len(lab)
    nc = 0
    for s in np.flatnonzero(pad.ravel() != 255).tolist():
        if comp[s] >= 0:
            continue
        l = lab[s]
        comp[s] = nc
        pila = [s]
        while pila:
            i = pila.pop()
            for j in (i - 1, i + 1, i - W, i + W):
                if lab[j] == l and comp[j] < 0:
                    comp[j] = nc
                    pila.append(j)
        nc += 1
    return np.array(comp, dtype=np.int32).reshape(h + 2, W)[1:-1, 1:-1], nc


def rellenar_huecos_selectivo(m, es_fondo, area_min):
    """Rellena los huecos internos de la prenda (botones, estampados, costuras), pero deja
    abiertos los huecos grandes cuyo contenido es fondo (p. ej. el espacio entre las piernas
    de un pantalón). es_fondo: máscara booleana de píxeles que K-means asignó a clusters de fondo."""
    lleno = rellenar_huecos(m)
    huecos = lleno & ~m
    if not huecos.any():
        return m
    comp, nc = componentes_por_etiqueta(np.where(huecos, 0, 255).astype(np.uint8))
    sel = comp >= 0
    area = np.bincount(comp[sel], minlength=nc)
    fondo = np.bincount(comp[sel], weights=es_fondo[sel].astype(np.float64), minlength=nc)
    mantener = (area >= area_min) & (fondo > 0.5 * area)
    out = lleno.copy()
    out[sel & mantener[np.maximum(comp, 0)]] = False
    return out


def mascara_por_regiones(et, k, suavizado, nucleo, cierre, ventana, umbral_centro,
                         area_min=0.002):
    """
    Decide qué es prenda y qué es fondo con las etiquetas de K-means:
      1) filtro de moda: cada píxel toma la etiqueta más frecuente de su vecindad
         (borra el moteado de pisos, telas con textura, etc.)
      2) núcleo: solo se conservan píxeles cuya vecindad es de una sola etiqueta
         (corta los puentes finos entre prenda y fondo del mismo color)
      3) semilla = clusters sobre-representados en la ventana central de la foto.
         "lift" = (parte del cluster que cae en el centro) / (parte de la imagen que es centro).
         El fondo tiene lift < 1 (está por todas partes); la prenda, lift alto. Así la prenda
         puede tocar el borde de la foto (pantalones, buzos recortados) sin problema.
      4) crecimiento: se suman las regiones pegadas a la prenda que NO están conectadas con
         el borde de la foto (mangas o partes de otro tono que quedan lejos del centro)
      5) se recupera el ancho del núcleo, se cierran rayas/estampados, se queda con la
         región más grande y se rellenan huecos
    """
    h, w = et.shape
    L = max(h, w)
    r_moda = max(1, int(round(suavizado * L)))
    r_nucleo = max(1, int(round(nucleo * L)))

    frac = np.stack([filtro_caja((et == j).astype(np.float64), r_moda) for j in range(k)])
    es = frac.argmax(axis=0)

    # Ventana central (ocupa una fracción `ventana` del ancho y del alto)
    y0, y1 = int(round(h * (1 - ventana) / 2)), int(round(h * (1 + ventana) / 2))
    x0, x1 = int(round(w * (1 - ventana) / 2)), int(round(w * (1 + ventana) / 2))
    en_centro = np.bincount(es[y0:y1, x0:x1].ravel(), minlength=k)
    total = np.maximum(np.bincount(es.ravel(), minlength=k), 1)
    lift = (en_centro / total) / ((y1 - y0) * (x1 - x0) / (h * w))
    es_prenda = lift >= umbral_centro
    if not es_prenda.any():
        es_prenda[np.argmax(lift)] = True
    print(f"Lift de cada cluster en el centro: {lift.round(2).tolist()}  "
          f"-> prenda: {np.flatnonzero(es_prenda).tolist()}")

    # Núcleo de todas las etiquetas (255 = barrera) y sus componentes conexas
    core = np.full((h, w), 255, dtype=np.uint8)
    for j in range(k):
        pureza = filtro_caja((es == j).astype(np.float64), r_nucleo)
        core[(es == j) & (pureza >= 0.999)] = j
    comp, nc = componentes_por_etiqueta(core)

    b = max(1, int(round(0.02 * min(h, w))))
    franja = np.ones((h, w), dtype=bool)
    franja[b:-b, b:-b] = False
    ids_borde = set(np.unique(comp[franja]).tolist()) - {-1}
    area = np.bincount(comp[comp >= 0], minlength=nc)
    etiqueta_comp = np.zeros(nc, dtype=np.int32)
    ys, xs = np.nonzero(comp >= 0)
    etiqueta_comp[comp[ys, xs]] = core[ys, xs]

    en_prenda = np.array([es_prenda[etiqueta_comp[c]] and area[c] > 0 for c in range(nc)])
    # Cluster del fondo dominante = el de la región más grande conectada con el borde.
    # Sus regiones interiores (p. ej. el espacio entre las piernas de un pantalón) son
    # fondo "encerrado": no deben sumarse a la prenda por crecimiento.
    borde_no_prenda = [c for c in ids_borde if not en_prenda[c]]
    etiqueta_fondo = (etiqueta_comp[max(borde_no_prenda, key=lambda c: area[c])]
                      if borde_no_prenda else -1)
    candidatos = [c for c in range(nc) if not en_prenda[c] and c not in ids_borde
                  and area[c] >= area_min * h * w and etiqueta_comp[c] != etiqueta_fondo]
    n_semilla = int(en_prenda.sum())
    # crecimiento: sumar candidatos tocando la prenda actual (hasta 3 pasadas)
    for _ in range(3):
        actual = dilatar_octogono(np.isin(comp, np.flatnonzero(en_prenda)), r_nucleo * 2 + 1)
        vecinos = set(np.unique(comp[actual & (comp >= 0)]).tolist())
        nuevos = [c for c in candidatos if c in vecinos and not en_prenda[c]]
        if not nuevos:
            break
        en_prenda[nuevos] = True
    print(f"Regiones: {nc}  (semilla: {n_semilla}, sumadas por crecimiento: "
          f"{int(en_prenda.sum()) - n_semilla})")

    prenda = np.isin(comp, np.flatnonzero(en_prenda))
    prenda = dilatar_octogono(prenda, r_nucleo)              # recuperar lo que se comió el núcleo
    prenda = cerrar_caja(prenda, int(round(cierre * L)))     # unir rayas / estampados
    # Proteger el fondo encerrado (espacio entre piernas, etc.): el cierre no debe taparlo
    encerradas = [c for c in range(nc) if not en_prenda[c] and c not in ids_borde
                  and etiqueta_comp[c] == etiqueta_fondo and area[c] >= 0.008 * h * w]
    if encerradas:
        prenda &= ~dilatar_octogono(np.isin(comp, encerradas), max(0, r_nucleo - 1))
    if not prenda.any():
        return None, es
    prenda = componente_mas_grande(prenda)
    prenda = rellenar_huecos_selectivo(prenda, ~es_prenda[es], 0.003 * h * w)
    return prenda, es


def segmentar(ruta, k=4, lado_trabajo=800, peso_espacial=0.0, peso_textura=3.0,
              cierre=0.03, suavizado=0.012, nucleo=0.004, ventana=0.4, umbral_centro=1.3):
    img = Image.open(ruta).convert("RGB")
    W0, H0 = img.size
    esc = min(1.0, lado_trabajo / max(W0, H0))
    if esc < 1.0:
        pequena = img.resize((max(1, int(W0 * esc)), max(1, int(H0 * esc))), Image.BILINEAR)
    else:
        pequena = img
    rgb = np.asarray(pequena, dtype=np.float64)
    h, w, _ = rgb.shape

    # Características por píxel: Lab (+ textura local, + posición, opcionales)
    lab = rgb_a_lab(rgb.reshape(-1, 3))
    columnas = [lab]
    if peso_textura > 0:
        tex = textura_local(lab[:, 0].reshape(h, w), r=3)
        columnas.append(peso_textura * tex.reshape(-1, 1))
    if peso_espacial > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        pos = np.stack([yy.ravel() / h, xx.ravel() / w], axis=1) * 100 * peso_espacial
        columnas.append(pos)
    feats = np.hstack(columnas)

    etiquetas, centros, iters = kmeans(feats, k)
    etiquetas = etiquetas.reshape(h, w)
    print(f"K-means convergió en {iters} iteraciones (k={k})")

    mascara, _ = mascara_por_regiones(etiquetas, k, suavizado, nucleo, cierre,
                                      ventana, umbral_centro)
    info = {"etiquetas": etiquetas, "rgb": rgb, "k": k}
    if mascara is None or not mascara.any():
        return img, None, info

    # Volver a la resolución original
    if esc < 1.0:
        m_img = Image.fromarray((mascara * 255).astype(np.uint8)).resize((W0, H0), Image.NEAREST)
        mascara_full = np.asarray(m_img) > 127
    else:
        mascara_full = mascara
    return img, mascara_full, info


def imagen_clusters(info):
    """Cada píxel pintado con el color medio de su cluster (para ver qué agrupó K-means)."""
    et, rgb = info["etiquetas"], info["rgb"]
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for j in range(info["k"]):
        sel = et == j
        if sel.any():
            out[sel] = rgb[sel].mean(axis=0).round().astype(np.uint8)
    return Image.fromarray(out)


def colores_dominantes(img, mascara, k=2):
    """K-means (en Lab) solo sobre los píxeles de la prenda."""
    rgb = np.asarray(img, dtype=np.float64)[mascara]
    paso = max(1, len(rgb) // 40000)  # submuestreo para que sea rápido
    muestra = rgb[::paso]
    etiquetas, _, _ = kmeans(rgb_a_lab(muestra), k)
    print("\n--- Colores dominantes de la prenda ---")
    orden = np.argsort(-np.bincount(etiquetas, minlength=k))
    for j in orden:
        miembros = muestra[etiquetas == j]
        if len(miembros) == 0:
            continue
        color = miembros.mean(axis=0).round().astype(int)
        print(f"RGB {color.tolist()}  #{color[0]:02x}{color[1]:02x}{color[2]:02x}  "
              f"-> {100 * len(miembros) / len(muestra):.1f}%")


def caracterizar(img, mascara):
    rgb = np.asarray(img, dtype=np.float64)
    px = rgb[mascara]
    ys, xs = np.nonzero(mascara)
    ancho, alto = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    print("\n--- Características de la prenda ---")
    print(f"Área (px):            {mascara.sum()}  ({100 * mascara.mean():.1f}% de la imagen)")
    print(f"Caja envolvente:      x[{xs.min()}-{xs.max()}]  y[{ys.min()}-{ys.max()}]")
    print(f"Relación alto/ancho:  {alto / ancho:.2f}")
    print(f"Color medio RGB:      {px.mean(axis=0).round(1).tolist()}")
    print(f"Desv. estándar RGB:   {px.std(axis=0).round(1).tolist()}")
    print(f"Centroide:            ({xs.mean():.1f}, {ys.mean():.1f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("imagen")
    ap.add_argument("--k", type=int, default=4, help="número de clusters")
    ap.add_argument("--textura", type=float, default=3.0,
                    help="peso de la textura local como característica (0 = solo color). "
                         "Ayuda cuando prenda y fondo tienen el mismo color pero distinta textura")
    ap.add_argument("--espacial", type=float, default=0.0,
                    help="peso de la posición (x,y) en las características, ej. 0.3")
    ap.add_argument("--cierre", type=float, default=0.03,
                    help="radio del cierre morfológico como fracción del lado mayor "
                         "(une rayas/estampados; 0 = desactivado)")
    ap.add_argument("--suavizado", type=float, default=0.012,
                    help="radio del filtro de moda (fracción del lado mayor). Más alto = "
                         "ignora texturas/moteado más grandes del fondo")
    ap.add_argument("--nucleo", type=float, default=0.004,
                    help="grosor mínimo de las regiones (fracción del lado mayor). Más alto = "
                         "corta uniones más gruesas entre prenda y fondo parecidos")
    ap.add_argument("--lado", type=int, default=800,
                    help="lado máximo de trabajo en px (más alto = bordes más finos, más lento)")
    ap.add_argument("--ventana", type=float, default=0.4,
                    help="tamaño de la ventana central (fracción del ancho y del alto) donde se "
                         "busca la prenda. Súbela si la prenda está muy descentrada")
    ap.add_argument("--umbral-centro", type=float, default=1.3,
                    help="un cluster es prenda si aparece en el centro al menos esta cantidad de "
                         "veces más que en el resto de la foto")
    ap.add_argument("--k-color", type=int, default=2,
                    help="colores dominantes a buscar dentro de la prenda")
    ap.add_argument("--salida", default=".", help="carpeta donde guardar los resultados")
    args = ap.parse_args()

    img, mascara, info = segmentar(args.imagen, k=args.k, lado_trabajo=args.lado,
                                   peso_espacial=args.espacial, peso_textura=args.textura,
                                   cierre=args.cierre, suavizado=args.suavizado,
                                   nucleo=args.nucleo, ventana=args.ventana,
                                   umbral_centro=args.umbral_centro)

    os.makedirs(args.salida, exist_ok=True)
    base = os.path.join(args.salida, os.path.splitext(os.path.basename(args.imagen))[0])
    if mascara is None:
        imagen_clusters(info).save(base + "_clusters.png")
        raise SystemExit(f"No se encontró ninguna prenda. Revisa {base}_clusters.png y prueba "
                         "otro --k, --suavizado 0.02 o --ventana 0.6 si la prenda está descentrada.")
    Image.fromarray((mascara * 255).astype(np.uint8)).save(base + "_mascara.png")
    rgb = np.asarray(img).copy()
    rgb[~mascara] = 0  # fuera de la prenda: transparente y sin color
    Image.fromarray(np.dstack([rgb, (mascara * 255).astype(np.uint8)]), "RGBA").save(base + "_sin_fondo.png")
    imagen_clusters(info).save(base + "_clusters.png")
    print(f"Guardados: {base}_mascara.png, {base}_sin_fondo.png, {base}_clusters.png")

    caracterizar(img, mascara)
    colores_dominantes(img, mascara, k=args.k_color)

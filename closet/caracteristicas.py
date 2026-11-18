"""Paso 4: características de la prenda ya normalizada (color, forma y textura).

Todo se calcula con NumPy sobre el resultado del paso 3 (`ResultadoNormalizacion`):

  * COLOR   K-means (el mismo del paso 2) sobre los píxeles del *núcleo* de la prenda, en Lab. El núcleo es la
            máscara erosionada, así el borde contaminado por el fondo no mete colores falsos. La imagen que llega
            ya tiene el color ORIGINAL (sin balance de blancos ni contraste), como se decidió en el paso 3.
  * FORMA   proporciones, solidez, anchura por alturas, número de tramos por altura (los pantalones se parten en
            dos piernas, las camisas tienen mangas) y los 7 momentos de Hu. Es lo que decidirá el tipo de prenda.
  * TEXTURA las 13 características de Haralick sobre la matriz de co-ocurrencia de niveles de gris, promediadas en
            4 direcciones. Haralick mide textura, no forma: sirve para describir el estampado (liso, rayas, cuadros).

Notas de diseño:
  - La matriz de co-ocurrencia solo cuenta pares donde AMBOS píxeles están dentro del núcleo (el fondo negro del
    lienzo nunca entra).
  - Los niveles de gris se reparten sobre 0-255 (no sobre el rango de cada prenda): así una prenda lisa sigue
    siendo lisa en vez de amplificar su ruido. Limitación: una prenda muy oscura usa pocos niveles.
  - Logaritmos naturales en las entropías (el tamaño de las unidades solo escala esos números).
  - La "varianza de la suma" usa la media de la suma (f6), no la entropía, como corrige la literatura posterior a 1973.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CaracteristicasConfig
from .normalizacion import ResultadoNormalizacion
from .segmentacion import erosion, filtro_caja, kmeans, rgb_a_lab

# ------------------------------------------------------------------ color
PALETA = {
    "negro": (20, 20, 20), "gris oscuro": (80, 80, 80), "gris": (128, 128, 128), "gris claro": (190, 190, 190),
    "blanco": (245, 245, 245), "crema": (240, 230, 200), "beige": (215, 195, 160), "caqui": (180, 165, 110),
    "marrón": (110, 70, 40), "rojo": (200, 30, 40), "vino": (110, 20, 40), "naranja": (240, 130, 30),
    "mostaza": (200, 160, 40), "amarillo": (240, 210, 50), "verde": (50, 150, 70), "verde oliva": (110, 115, 50),
    "verde oscuro": (20, 70, 40), "turquesa": (40, 160, 160), "celeste": (130, 190, 235), "azul": (40, 80, 200),
    "azul marino": (25, 35, 85), "azul jean": (70, 100, 140), "jean claro": (140, 165, 195),
    "morado": (110, 50, 150), "rosa": (240, 140, 180), "fucsia": (210, 40, 140),
}
_NOMBRES = list(PALETA)
_LAB_PALETA = rgb_a_lab(np.array(list(PALETA.values()), dtype=np.float64))


def nombre_color(lab: np.ndarray) -> str:
    """Nombre en español del color de la paleta más cercano (distancia euclidiana en Lab, ΔE76)."""
    lab = np.asarray(lab, dtype=np.float64)
    L, croma, tono = lab[0], float(np.hypot(lab[1], lab[2])), float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)
    # Reglas para los casos donde la paleta falla (los colores oscuros y los blancos con luz de cuarto quedan muy
    # cerca entre sí en Lab): un negro con un tinte azul claro es azul marino, uno con tinte cálido es marrón.
    if L >= 78 and croma < 8:
        return "blanco"
    if L < 25 and croma >= 8 and 220 <= tono <= 300:
        return "azul marino"
    if L < 25 and croma >= 4.5 and 15 <= tono <= 80:
        return "marrón"
    d = np.linalg.norm(_LAB_PALETA - lab[None], axis=1)
    return _NOMBRES[int(np.argmin(d))]


def _mismo_color_bajo_sombra(a, b):
    """Compara tonos Lab separando luminosidad de dirección cromática."""
    c1, c2 = np.linalg.norm(a[1:]), np.linalg.norm(b[1:])
    if max(c1, c2) < 10:
        return abs(a[0] - b[0]) < 40
    if min(c1, c2) < 7:
        return False
    coseno = float(np.dot(a[1:], b[1:]) / (c1 * c2))
    return coseno >= .96 and (abs(c1-c2) <= 10 or
                                (abs(a[0]-b[0]) >= 12 and min(c1,c2)/max(c1,c2) >= .62))


def caracteristicas_color(rgb: np.ndarray, nucleo: np.ndarray, cfg: CaracteristicasConfig | None = None) -> dict:
    """Color dominante, segundo color, claridad, saturación y cuántos colores distintos tiene la prenda."""
    cfg = cfg or CaracteristicasConfig()
    px = rgb[nucleo].astype(np.float64)
    if len(px) == 0:
        raise ValueError("el núcleo de la prenda está vacío")
    if len(px) > cfg.max_pixeles_color:                         # submuestreo regular (determinista)
        px = px[np.linspace(0, len(px) - 1, cfg.max_pixeles_color).astype(int)]
    lab = rgb_a_lab(px)
    k = max(1, min(cfg.k_color, len(np.unique(px, axis=0))))
    etiquetas, centros, _ = kmeans(lab, k, seed=cfg.semilla)
    frac = np.bincount(etiquetas, minlength=k) / len(etiquetas)
    orden = np.argsort(-frac)
    rgb_cluster = [px[etiquetas == j].mean(axis=0) for j in orden]

    significativos = []                                          # colores que cubren algo y se distinguen entre sí
    for j in orden:
        if frac[j] < cfg.fraccion_color_min and significativos:
            continue
        if all(np.linalg.norm(centros[j] - centros[i]) >= cfg.distancia_color_min for i in significativos):
            significativos.append(j)

    croma = np.hypot(lab[:, 1], lab[:, 2])
    cromaticos = []
    for j in orden:
        if frac[j] < cfg.fraccion_color_min and cromaticos:
            continue
        if all(not _mismo_color_bajo_sombra(centros[j], centros[i]) for i in cromaticos):
            cromaticos.append(j)
    r0 = np.rint(rgb_cluster[0]).astype(int)
    d = {
        "color_nombre": nombre_color(centros[orden[0]]),
        "color_r": int(r0[0]), "color_g": int(r0[1]), "color_b": int(r0[2]),
        "color_hex": "#%02x%02x%02x" % tuple(r0),
        "color_L": round(float(centros[orden[0]][0]), 2), "color_a": round(float(centros[orden[0]][1]), 2),
        "color_b_lab": round(float(centros[orden[0]][2]), 2),
        "color_frac": round(float(frac[orden[0]]), 3),
        "claridad_media": round(float(lab[:, 0].mean()), 2),
        "croma_medio": round(float(croma.mean()), 2),
        "n_colores": len(significativos),
        "n_colores_cromaticos": len(cromaticos),
    }
    if len(orden) > 1:
        r1 = np.rint(rgb_cluster[1]).astype(int)
        d.update(color2_nombre=nombre_color(centros[orden[1]]), color2_hex="#%02x%02x%02x" % tuple(r1),
                 color2_frac=round(float(frac[orden[1]]), 3))
    else:
        d.update(color2_nombre="", color2_hex="", color2_frac=0.0)
    return d


# ------------------------------------------------------------------ forma
def momentos_hu(mascara: np.ndarray) -> np.ndarray:
    """Los 7 momentos invariantes de Hu de la máscara, en escala logarítmica (-sign(h)·log10|h|)."""
    ys, xs = np.nonzero(mascara)
    dx, dy = xs - xs.mean(), ys - ys.mean()
    m00 = float(len(xs))

    def eta(p, q):
        return float((dx ** p * dy ** q).sum()) / m00 ** (1 + (p + q) / 2)

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n03, n21, n12 = eta(3, 0), eta(0, 3), eta(2, 1), eta(1, 2)
    h = np.array([
        n20 + n02,
        (n20 - n02) ** 2 + 4 * n11 ** 2,
        (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
        (n30 + n12) ** 2 + (n21 + n03) ** 2,
        (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03),
        (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
    ])
    return -np.sign(h) * np.log10(np.maximum(np.abs(h), 1e-30))


def _casco_convexo(puntos: np.ndarray) -> np.ndarray:
    """Casco convexo (cadena monótona de Andrew). puntos: (N,2) -> vértices en orden."""
    pts = sorted(set(map(tuple, puntos.tolist())))
    if len(pts) <= 2:
        return np.array(pts, dtype=np.float64)

    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    mitad = []
    for secuencia in (pts, pts[::-1]):
        cadena = []
        for p in secuencia:
            while len(cadena) >= 2 and cruz(cadena[-2], cadena[-1], p) <= 0:
                cadena.pop()
            cadena.append(p)
        mitad.append(cadena[:-1])
    return np.array(mitad[0] + mitad[1], dtype=np.float64)


def area_casco(mascara: np.ndarray) -> float:
    """Área del casco convexo de la máscara (en píxeles, contando el píxel completo, no solo su centro)."""
    ys, xs = np.nonzero(mascara)
    puntos = []
    for y in np.unique(ys):
        fila = xs[ys == y]
        l, r = fila.min() - 0.5, fila.max() + 0.5
        puntos += [(l, y - 0.5), (l, y + 0.5), (r, y - 0.5), (r, y + 0.5)]
    h = _casco_convexo(np.array(puntos))
    if len(h) < 3:
        return float(mascara.sum())
    x, y = h[:, 0], h[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _tramos(fila: np.ndarray, minimo: int) -> int:
    """Cuántos tramos continuos de `True` de al menos `minimo` píxeles hay en la fila."""
    d = np.diff(np.concatenate([[0], fila.astype(np.int8), [0]]))
    inicios, fines = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return int(((fines - inicios) >= minimo).sum())


def caracteristicas_forma(mascara: np.ndarray, cfg: CaracteristicasConfig | None = None) -> dict:
    """Proporciones, solidez, anchura por alturas, tramos por altura, simetría y momentos de Hu."""
    cfg = cfg or CaracteristicasConfig()
    if not mascara.any():
        raise ValueError("la máscara está vacía")
    ys, xs = np.nonzero(mascara)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    caja = mascara[y0:y1 + 1, x0:x1 + 1]
    alto, ancho = caja.shape
    area = int(mascara.sum())

    # perímetro como caras expuestas; se compara con el de un rectángulo del mismo tamaño (1.0 = sin entrantes)
    p = np.pad(caja, 1)
    bordes = int((p[1:-1, 1:-1] & ~p[:-2, 1:-1]).sum() + (p[1:-1, 1:-1] & ~p[2:, 1:-1]).sum()
                 + (p[1:-1, 1:-1] & ~p[1:-1, :-2]).sum() + (p[1:-1, 1:-1] & ~p[1:-1, 2:]).sum())

    d = {
        "ratio_aspecto": round(alto / ancho, 4),
        "extension": round(area / (alto * ancho), 4),
        "solidez": round(min(1.0, area / area_casco(mascara)), 4),
        "perimetro_rel": round(bordes / (2 * (alto + ancho)), 4),
        "simetria": round(float((caja & caja[:, ::-1]).sum() / (caja | caja[:, ::-1]).sum()), 4),
        "fraccion_superior": round(float(caja[: alto // 2].sum() / area), 4),
    }
    minimo = max(1, int(round(cfg.tramo_min * ancho)))
    for i in range(cfg.n_alturas):
        fila = caja[min(alto - 1, int((i + 0.5) / cfg.n_alturas * alto))]
        cols = np.nonzero(fila)[0]
        span = (cols.max() - cols.min() + 1) / ancho if len(cols) else 0.0
        d[f"anchura_{i}"] = round(float(span), 4)
        d[f"relleno_{i}"] = round(float(fila.sum() / ancho), 4)
        d[f"tramos_{i}"] = _tramos(fila, minimo) if len(cols) else 0
    for i, h in enumerate(momentos_hu(mascara), start=1):
        d[f"hu{i}"] = round(float(h), 4)
    return d


# ---------------------------------------------------------------- textura
NOMBRES_HARALICK = ["energia", "contraste", "correlacion", "varianza", "homogeneidad", "suma_media",
                    "suma_varianza", "suma_entropia", "entropia", "dif_varianza", "dif_entropia", "imc1", "imc2"]

# (dy, dx) de cada dirección: 0°, 45°, 90°, 135° (y crece hacia abajo)
DIRECCIONES = {0: (0, 1), 45: (-1, 1), 90: (-1, 0), 135: (-1, -1)}


def luminancia(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def cuantizar(gris: np.ndarray, niveles: int) -> np.ndarray:
    """Reparte 0-255 en `niveles` partes iguales -> enteros 0..niveles-1."""
    return np.minimum((gris * niveles / 256.0).astype(np.int64), niveles - 1)


def matriz_coocurrencia(q: np.ndarray, valido: np.ndarray, dy: int, dx: int, niveles: int) -> np.ndarray | None:
    """GLCM simétrica y normalizada de los pares (píxel, píxel + (dy, dx)) con ambos dentro de `valido`.

    Devuelve None si no hay ningún par (prenda demasiado pequeña para esa distancia).
    """
    alto, ancho = q.shape
    ya, yb = max(0, -dy), alto - max(0, dy)
    xa, xb = max(0, -dx), ancho - max(0, dx)
    a, b = q[ya:yb, xa:xb], q[ya + dy:yb + dy, xa + dx:xb + dx]
    ok = valido[ya:yb, xa:xb] & valido[ya + dy:yb + dy, xa + dx:xb + dx]
    if not ok.any():
        return None
    cuentas = np.bincount(a[ok] * niveles + b[ok], minlength=niveles * niveles).reshape(niveles, niveles)
    glcm = (cuentas + cuentas.T).astype(np.float64)
    return glcm / glcm.sum()


def _entropia(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def haralick(P: np.ndarray) -> dict:
    """Las 13 características de Haralick (1973) de una GLCM simétrica y normalizada."""
    N = P.shape[0]
    i = np.arange(N, dtype=np.float64)
    I, J = np.meshgrid(i, i, indexing="ij")
    px = P.sum(axis=1)
    py = P.sum(axis=0)
    mux, muy = (i * px).sum(), (i * py).sum()
    sx = np.sqrt(((i - mux) ** 2 * px).sum())
    sy = np.sqrt(((i - muy) ** 2 * py).sum())
    p_suma = np.bincount((I + J).astype(int).ravel(), weights=P.ravel(), minlength=2 * N - 1)
    p_dif = np.bincount(np.abs(I - J).astype(int).ravel(), weights=P.ravel(), minlength=N)
    k_s, k_d = np.arange(2 * N - 1, dtype=np.float64), np.arange(N, dtype=np.float64)

    suma_media = float((k_s * p_suma).sum())
    dif_media = float((k_d * p_dif).sum())
    pxpy = np.outer(px, py)
    hxy, hx, hy = _entropia(P.ravel()), _entropia(px), _entropia(py)
    mascara = P > 0
    hxy1 = float(-(P[mascara] * np.log(np.maximum(pxpy[mascara], 1e-300))).sum())
    pp = pxpy[pxpy > 0]
    hxy2 = float(-(pp * np.log(pp)).sum())
    return {
        "energia": float((P ** 2).sum()),
        "contraste": float((k_d ** 2 * p_dif).sum()),
        "correlacion": float(((I * J * P).sum() - mux * muy) / (sx * sy)) if sx * sy > 1e-12 else 1.0,
        "varianza": float(((I - mux) ** 2 * P).sum()),
        "homogeneidad": float((P / (1 + (I - J) ** 2)).sum()),
        "suma_media": suma_media,
        "suma_varianza": float(((k_s - suma_media) ** 2 * p_suma).sum()),
        "suma_entropia": _entropia(p_suma),
        "entropia": hxy,
        "dif_varianza": float(((k_d - dif_media) ** 2 * p_dif).sum()),
        "dif_entropia": _entropia(p_dif),
        "imc1": float((hxy - hxy1) / max(hx, hy)) if max(hx, hy) > 1e-12 else 0.0,
        "imc2": float(np.sqrt(max(0.0, 1 - np.exp(-2 * (hxy2 - hxy))))),
    }


def _promedio_en_nucleo(x: np.ndarray, m: np.ndarray, r: int) -> np.ndarray:
    """Promedio local de un mapa (alto, ancho, canales) usando solo los píxeles de `m` (no mezcla el fondo negro)."""
    num = np.stack([filtro_caja(x[..., k] * m, r) for k in range(x.shape[2])], axis=-1)
    return num / np.maximum(filtro_caja(m, r)[..., None], 1e-9)


def detalle_sin_sombras(rgb: np.ndarray, nucleo: np.ndarray, cfg: CaracteristicasConfig | None = None) -> float:
    """Fracción de la prenda con detalle de color/brillo que NO es sombra ni pliegue (0 = liso).

    Pasa-banda en Lab: (promedio en ventana chica) - (promedio en ventana grande). La ventana grande es la
    iluminación y los pliegues suaves; lo que queda es el contraste fino (estampados, rayas, logos, costuras).
    Se mide la fracción de píxeles cuya diferencia supera `detalle_umbral` (ΔE) y se ignora el borde del
    núcleo, donde la máscara misma crea contrastes falsos. Sirve también para prendas oscuras: un estampado gris
    sobre negro tiene poco Haralick (casi todo cae en un solo nivel de gris) pero sí un ΔE alto.
    """
    cfg = cfg or CaracteristicasConfig()
    h, w, _ = rgb.shape
    lab = rgb_a_lab(rgb.reshape(-1, 3).astype(np.float64)).reshape(h, w, 3)
    m = nucleo.astype(np.float64)
    banda = np.linalg.norm(_promedio_en_nucleo(lab, m, cfg.detalle_radio_chico)
                           - _promedio_en_nucleo(lab, m, cfg.detalle_radio_grande), axis=2)
    interior = erosion(nucleo, cfg.detalle_borde)
    if not interior.any():
        interior = nucleo
    return float((banda[interior] > cfg.detalle_umbral).mean())


def caracteristicas_textura(rgb: np.ndarray, nucleo: np.ndarray, cfg: CaracteristicasConfig | None = None) -> dict:
    """Haralick promediado en 4 direcciones (distancia `distancia_haralick`) y medidas de patrón a mayor distancia.

    Además del Haralick estándar devuelve, a `distancia_patron`: el contraste en cada dirección y la anisotropía
    (cuánto más varía la prenda en horizontal que en vertical). Rayas verticales -> contraste horizontal alto y
    vertical bajo; cuadros -> ambos altos; liso -> ambos bajos.
    """
    cfg = cfg or CaracteristicasConfig()
    q = cuantizar(luminancia(rgb), cfg.niveles_gris)
    por_dir = {}
    for ang, (dy, dx) in DIRECCIONES.items():
        P = matriz_coocurrencia(q, nucleo, dy * cfg.distancia_haralick, dx * cfg.distancia_haralick, cfg.niveles_gris)
        if P is not None:
            por_dir[ang] = haralick(P)
    if not por_dir:
        raise ValueError("la prenda es demasiado pequeña para medir su textura")
    d = {f"har_{n}": round(float(np.mean([h[n] for h in por_dir.values()])), 5) for n in NOMBRES_HARALICK}

    c = {}
    for ang, (dy, dx) in DIRECCIONES.items():
        P = matriz_coocurrencia(q, nucleo, dy * cfg.distancia_patron, dx * cfg.distancia_patron, cfg.niveles_gris)
        c[ang] = haralick(P)["contraste"] if P is not None else 0.0
    ch, cv = c[0], c[90]
    d["detalle_frac"] = round(detalle_sin_sombras(rgb, nucleo, cfg), 4)
    d.update({
        "patron_contraste_h": round(ch, 4), "patron_contraste_v": round(cv, 4),
        "patron_contraste_d": round((c[45] + c[135]) / 2, 4),
        "patron_anisotropia": round((ch - cv) / (ch + cv), 4) if ch + cv > 1e-9 else 0.0,
    })
    return d


def describir_estampado(textura: dict, cfg: CaracteristicasConfig | None = None) -> str:
    """"liso" o "con_textura". Conservador a propósito: solo dice "liso" cuando todas las medidas coinciden.

    Con las medidas de Haralick NO se pueden separar de forma fiable rayas y cuadros de pliegues, costuras o
    acolchado (un pantalón liso con pliegues verticales marca igual de anisotrópico que una camisa de rayas),
    así que no se inventa esa distinción: "con_textura" agrupa estampados, rayas, cuadros, logos y relieve.
    """
    cfg = cfg or CaracteristicasConfig()
    if "detalle_frac" in textura:           # medida sin sombras: no confunde pliegues con estampado
        return "liso" if textura["detalle_frac"] < cfg.detalle_liso_max else "con_textura"
    liso = (textura["har_homogeneidad"] >= cfg.liso_homogeneidad_min
            and textura["har_contraste"] < cfg.liso_contraste_max
            and textura["patron_contraste_d"] < cfg.liso_patron_max)
    return "liso" if liso else "con_textura"


# ---------------------------------------------------------------- conjunto
@dataclass
class Caracteristicas:
    color: dict = field(default_factory=dict)
    forma: dict = field(default_factory=dict)
    textura: dict = field(default_factory=dict)

    def como_dict(self) -> dict:
        """Todo en un solo diccionario plano (una fila de CSV)."""
        return {**self.color, **self.forma, **self.textura}


def extraer_caracteristicas(norm: ResultadoNormalizacion, cfg: CaracteristicasConfig | None = None) -> Caracteristicas:
    """Color, forma y textura de una prenda normalizada (resultado del paso 3)."""
    cfg = cfg or CaracteristicasConfig()
    nucleo = norm.nucleo if norm.nucleo.any() else norm.mascara     # prendas muy delgadas: se usa la máscara entera
    textura = caracteristicas_textura(norm.rgb, nucleo, cfg)
    textura["estampado"] = describir_estampado(textura, cfg)
    return Caracteristicas(
        color=caracteristicas_color(norm.rgb, nucleo, cfg),
        forma=caracteristicas_forma(norm.mascara, cfg),
        textura=textura,
    )


def columnas_forma(cfg: CaracteristicasConfig | None = None) -> list[str]:
    """Nombres de las características de forma numéricas, en orden fijo (para el clasificador del paso 5)."""
    cfg = cfg or CaracteristicasConfig()
    base = ["ratio_aspecto", "extension", "solidez", "perimetro_rel", "simetria", "fraccion_superior"]
    por_altura = [f"{n}_{i}" for i in range(cfg.n_alturas) for n in ("anchura", "relleno", "tramos")]
    return base + por_altura + [f"hu{i}" for i in range(1, 8)]

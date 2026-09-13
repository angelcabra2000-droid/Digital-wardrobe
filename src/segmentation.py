"""
segmentation.py
-----------------
Paso 3 del pipeline: Segmentación de la prenda.

Recibe la imagen preprocesada y la máscara de región válida que produce
preprocessing.py (image, mask = preprocess_image(...)) y devuelve una
máscara booleana final que marca los píxeles que pertenecen a la
prenda.

Criterio de "pureza" (igual que en preprocessing.py):
- cv2 se usa SOLO para infraestructura: conversión de espacio de color
  (RGB -> escala de grises, RGB -> HSV). Esas conversiones son
  operaciones de codificación de color, no el algoritmo de
  segmentación que evalúa el curso.
- El umbral de Otsu (histograma + maximización de varianza
  entre-clases) y la limpieza por componente conexo (BFS) están
  implementados a mano con numpy, sin cv2.threshold, cv2.THRESH_OTSU
  ni cv2.connectedComponents.

Por qué se prueban 4 máscaras candidatas:
No sabemos de antemano si la prenda es más clara o más oscura que el
fondo, ni si el canal de gris o el de brillo (V de HSV) separa mejor
esa foto en particular. En vez de asumirlo, se calcula Otsu sobre dos
canales (gris y V) y se evalúan las dos polaridades de cada uno
(prenda-clara-sobre-fondo-oscuro y viceversa). Eso da 4 candidatas:

    (gris,  prenda_clara)
    (gris,  prenda_oscura)
    (v,     prenda_clara)
    (v,     prenda_oscura)

Cada candidata se puntúa con 3 criterios (ver _score_mask) y se elige
la de mayor puntaje. Sobre esa ganadora se aplica limpieza final
quedándose con el componente conexo más grande (BFS a mano), para
eliminar salpicaduras de ruido que hayan quedado con el mismo signo
del umbral pero sin ser parte de la prenda.

Nota sobre la máscara de región válida (mask):
El relleno negro del letterbox (ver preprocessing.py) NO es parte de
la foto real. Si se dejara pasar como si fuera background legítimo,
crearía un pico artificial enorme en el histograma que rompe Otsu
(el umbral terminaría separando "negro de relleno vs. todo lo demás"
en lugar de "prenda vs. fondo real"). Por eso, en este archivo:

1. El histograma de Otsu se calcula ÚNICAMENTE sobre píxeles donde
   mask == True.
2. El área de cada candidata se mide como proporción del área VÁLIDA
   (mask.sum()), no del canvas completo.
3. El "border-touch" (qué tanto la candidata toca el borde de la foto
   real) se mide contra el borde de la región válida, no contra el
   borde del canvas -- si hay letterbox, el borde real de la foto
   puede estar en medio del canvas, no en sus bordes literales.
"""

import cv2
import numpy as np

# Pesos por defecto del puntaje combinado (ver _score_mask). Se dejan
# como constantes fáciles de ajustar si en la práctica un criterio
# resulta más confiable que otro para las fotos reales del proyecto.
PESO_AREA = 1.0
PESO_COMPACIDAD = 1.0
PESO_BORDE = 1.0

# Rango de proporción de área que se considera "razonable" para una
# prenda extendida y flat-lay sobre un fondo que la rodea (protocolo
# de captura: fondo contrastante, prenda plana). Fuera de este rango
# se penaliza porque probablemente el umbral quedó invertido (segmentó
# el fondo) o el ruido dominó la imagen.
AREA_PROPORCION_MIN = 0.05
AREA_PROPORCION_MAX = 0.85


def _otsu_threshold(channel: np.ndarray, valid_mask: np.ndarray) -> int:
    """
    Calcula el umbral de Otsu a mano sobre un solo canal (0-255),
    considerando SOLO los píxeles donde valid_mask == True.

    Implementación clásica por maximización de varianza entre-clases:
    1. Histograma de 256 niveles (solo píxeles válidos).
    2. Para cada umbral candidato t, separar en clase "fondo" (<=t) y
       clase "primer plano" (>t).
    3. Calcular la varianza entre-clases para ese t:
           sigma_b^2(t) = w_bg(t) * w_fg(t) * (mu_bg(t) - mu_fg(t))^2
    4. El umbral óptimo es el que maximiza sigma_b^2(t).

    Se usan sumas acumuladas (running sums) en vez de recalcular medias
    desde cero en cada t, para que el barrido de los 256 umbrales sea
    O(256) en vez de O(256 * n_pixeles).
    """
    valores = channel[valid_mask]
    histograma, _ = np.histogram(valores, bins=256, range=(0, 256))

    total_pixeles = valores.size
    suma_total = np.dot(np.arange(256), histograma)

    peso_fondo = 0
    suma_fondo = 0
    mejor_varianza = -1.0
    mejor_umbral = 0

    for t in range(256):
        peso_fondo += histograma[t]
        if peso_fondo == 0:
            continue

        peso_primer_plano = total_pixeles - peso_fondo
        if peso_primer_plano == 0:
            break

        suma_fondo += t * histograma[t]
        media_fondo = suma_fondo / peso_fondo
        media_primer_plano = (suma_total - suma_fondo) / peso_primer_plano

        varianza_entre_clases = (
            peso_fondo * peso_primer_plano * (media_fondo - media_primer_plano) ** 2
        )

        if varianza_entre_clases > mejor_varianza:
            mejor_varianza = varianza_entre_clases
            mejor_umbral = t

    return mejor_umbral


def _mascara_desde_umbral(
    channel: np.ndarray, umbral: int, valid_mask: np.ndarray, prenda_clara: bool
) -> np.ndarray:
    """
    Construye una máscara booleana candidata a partir de un umbral de
    Otsu, para una de las dos polaridades posibles:

    - prenda_clara=True:  la prenda es MÁS BRILLANTE que el fondo,
      así que el primer plano es channel > umbral.
    - prenda_clara=False: la prenda es MÁS OSCURA que el fondo,
      así que el primer plano es channel <= umbral.

    En ambos casos se intersecta con valid_mask: un píxel de relleno
    de letterbox nunca puede terminar marcado como prenda, sin
    importar de qué lado del umbral caiga.
    """
    if prenda_clara:
        candidata = channel > umbral
    else:
        candidata = channel <= umbral

    return candidata & valid_mask


def _perimetro(mask: np.ndarray) -> int:
    """
    Cuenta los píxeles de borde de una máscara booleana: un píxel
    True que tiene AL MENOS un vecino (4-conectividad) en False o que
    cae en el borde del array (se trata como si su vecino externo
    fuera False).

    Se usa desplazamiento de arrays (comparaciones vectorizadas) en
    vez de recorrer píxel por píxel, ya que esto se calcula una vez
    por cada una de las 4 candidatas y no es el algoritmo central que
    se evalúa (el corazón conceptual aquí es Otsu y la compacidad que
    se deriva de este perímetro, no el recorrido en sí).
    """
    arriba = np.zeros_like(mask)
    abajo = np.zeros_like(mask)
    izquierda = np.zeros_like(mask)
    derecha = np.zeros_like(mask)

    arriba[1:, :] = mask[:-1, :]
    abajo[:-1, :] = mask[1:, :]
    izquierda[:, 1:] = mask[:, :-1]
    derecha[:, :-1] = mask[:, 1:]

    tiene_los_4_vecinos_dentro = arriba & abajo & izquierda & derecha
    es_borde = mask & ~tiene_los_4_vecinos_dentro

    return int(es_borde.sum())


def _compacidad(mask: np.ndarray) -> float:
    """
    Índice isoperimétrico clásico de compacidad:

        compacidad = (4 * pi * area) / perimetro^2

    - Un círculo perfecto da compacidad ~= 1 (el máximo posible en el
      continuo; en una grilla discreta puede superar 1 ligeramente
      por el conteo de perímetro, por eso se recorta a 1.0).
    - Una máscara hecha pedazos (ruido salpicado por toda la imagen,
      típico de un umbral de Otsu mal orientado) tiene un perímetro
      enorme respecto a su área, así que la compacidad cae cerca de 0.

    Máscara vacía o de un solo píxel: se devuelve 0.0 (no hay forma
    que evaluar, y no queremos que una candidata vacía "gane" por
    división entre cero).
    """
    area = int(mask.sum())
    if area == 0:
        return 0.0

    perimetro = _perimetro(mask)
    if perimetro == 0:
        return 0.0

    valor = (4.0 * np.pi * area) / (perimetro ** 2)
    return float(min(valor, 1.0))


def _borde_de_region_valida(valid_mask: np.ndarray) -> np.ndarray:
    """
    Marca los píxeles que forman el borde REAL de la foto original,
    es decir, el contorno de la región válida (mask == True) que
    produce resize_image() en preprocessing.py.

    Esto NO es lo mismo que el borde del canvas: si hubo letterbox
    (por ejemplo una foto vertical rellenada a los lados), el borde
    real de la foto queda en medio del canvas, justo donde termina la
    región válida y empieza el relleno negro.
    """
    return valid_mask & ~_erosionar_una_capa(valid_mask)


def _erosionar_una_capa(mask: np.ndarray) -> np.ndarray:
    """
    Erosión binaria de 1 píxel con elemento estructurante en cruz
    (4-conectividad), implementada a mano con desplazamientos de
    arrays: un píxel sobrevive solo si él y sus 4 vecinos son True
    (los píxeles fuera del array cuentan como False).
    """
    arriba = np.zeros_like(mask)
    abajo = np.zeros_like(mask)
    izquierda = np.zeros_like(mask)
    derecha = np.zeros_like(mask)

    arriba[1:, :] = mask[:-1, :]
    abajo[:-1, :] = mask[1:, :]
    izquierda[:, 1:] = mask[:, :-1]
    derecha[:, :-1] = mask[:, 1:]

    return mask & arriba & abajo & izquierda & derecha


def _ratio_contacto_borde(candidata: np.ndarray, borde_valido: np.ndarray) -> float:
    """
    Proporción del borde real de la foto que quedó marcada como
    "prenda" en la candidata.

    Por el protocolo de captura (fondo que contrasta, rodeando a la
    prenda extendida), se espera que la prenda NO toque los bordes de
    la foto -- lo que toca los bordes normalmente es el fondo. Una
    candidata con un ratio alto aquí es señal de que el umbral quedó
    invertido (segmentó fondo, no prenda), así que se penaliza.
    """
    total_borde = int(borde_valido.sum())
    if total_borde == 0:
        return 0.0

    contacto = int((candidata & borde_valido).sum())
    return contacto / total_borde


def _score_mask(
    candidata: np.ndarray,
    valid_mask: np.ndarray,
    borde_valido: np.ndarray,
    area_valida_total: int,
) -> dict:
    """
    Combina los 3 criterios de evaluación de una candidata en un solo
    puntaje, y devuelve también los valores individuales (útiles para
    depurar por qué una candidata ganó o perdió).

    - area_score: 1.0 si la proporción de área (respecto al área
      VÁLIDA, no el canvas) cae dentro del rango razonable
      [AREA_PROPORCION_MIN, AREA_PROPORCION_MAX]; decae linealmente
      hacia 0 mientras más se aleje del rango.
    - compacidad: ver _compacidad().
    - borde_score: 1.0 - ratio_contacto_borde (menos contacto con el
      borde real de la foto es mejor).
    """
    area_candidata = int(candidata.sum())
    proporcion = area_candidata / area_valida_total if area_valida_total > 0 else 0.0

    if AREA_PROPORCION_MIN <= proporcion <= AREA_PROPORCION_MAX:
        area_score = 1.0
    elif proporcion < AREA_PROPORCION_MIN:
        area_score = proporcion / AREA_PROPORCION_MIN if AREA_PROPORCION_MIN > 0 else 0.0
    else:
        exceso = proporcion - AREA_PROPORCION_MAX
        area_score = max(0.0, 1.0 - exceso / (1.0 - AREA_PROPORCION_MAX))

    compacidad = _compacidad(candidata)
    ratio_borde = _ratio_contacto_borde(candidata, borde_valido)
    borde_score = 1.0 - ratio_borde

    puntaje_total = (
        PESO_AREA * area_score
        + PESO_COMPACIDAD * compacidad
        + PESO_BORDE * borde_score
    )

    return {
        "area_proporcion": proporcion,
        "area_score": area_score,
        "compacidad": compacidad,
        "ratio_contacto_borde": ratio_borde,
        "borde_score": borde_score,
        "puntaje_total": puntaje_total,
    }


def _componente_conexo_mas_grande(mask: np.ndarray) -> np.ndarray:
    """
    Limpieza final: de todos los componentes conexos (4-conectividad)
    de la máscara ganadora, se queda solo con el más grande y descarta
    el resto como ruido residual.

    Etiquetado por BFS implementado a mano (sin cv2.connectedComponents
    ni scipy.ndimage.label): se recorre cada píxel True no visitado,
    se expande por BFS a sus vecinos 4-conectados, y se cuenta el
    tamaño de cada componente encontrado.

    La cola de BFS se implementa con una lista + un puntero de lectura
    (en vez de list.pop(0), que es O(n) por llamada) para que cada
    componente se procese en O(tamaño_del_componente).
    """
    alto, ancho = mask.shape
    visitado = np.zeros_like(mask, dtype=bool)

    mejor_tamano = 0
    mejor_componente = np.zeros_like(mask, dtype=bool)

    for fila_inicio in range(alto):
        for col_inicio in range(ancho):
            if not mask[fila_inicio, col_inicio] or visitado[fila_inicio, col_inicio]:
                continue

            cola = [(fila_inicio, col_inicio)]
            visitado[fila_inicio, col_inicio] = True
            indice_lectura = 0
            pixeles_componente = []

            while indice_lectura < len(cola):
                fila, col = cola[indice_lectura]
                indice_lectura += 1
                pixeles_componente.append((fila, col))

                for df, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nf, nc = fila + df, col + dc
                    if 0 <= nf < alto and 0 <= nc < ancho:
                        if mask[nf, nc] and not visitado[nf, nc]:
                            visitado[nf, nc] = True
                            cola.append((nf, nc))

            if len(pixeles_componente) > mejor_tamano:
                mejor_tamano = len(pixeles_componente)
                mejor_componente = np.zeros_like(mask, dtype=bool)
                filas, cols = zip(*pixeles_componente)
                mejor_componente[list(filas), list(cols)] = True

    return mejor_componente


def segment_garment(
    image: np.ndarray, valid_mask: np.ndarray, debug: bool = False
) -> np.ndarray:
    """
    Segmenta la prenda dentro de `image` (RGB, salida de
    preprocess_image), usando `valid_mask` para ignorar el relleno de
    letterbox en todo el proceso.

    Pasos:
    1. Derivar dos canales candidatos: escala de grises y V (brillo)
       de HSV. Estas conversiones de espacio de color son
       infraestructura (cv2 permitido), no algoritmo del curso.
    2. Para cada canal, calcular su umbral de Otsu SOLO sobre píxeles
       válidos, y construir las 2 máscaras de polaridad (prenda clara
       / prenda oscura). Total: 4 candidatas.
    3. Puntuar cada candidata con _score_mask (área, compacidad,
       contacto con el borde real de la foto).
    4. Quedarse con la candidata de mayor puntaje.
    5. Limpiar esa candidata quedándose con su componente conexo más
       grande (BFS), para descartar ruido residual disperso.

    Si debug=True, devuelve además la lista de puntajes de las 4
    candidatas (útil para inspeccionar por qué una ganó sobre las
    otras al calibrar sobre fotos reales).
    """
    gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    valor_hsv = hsv[:, :, 2]

    area_valida_total = int(valid_mask.sum())
    borde_valido = _borde_de_region_valida(valid_mask)

    candidatas_info = []

    for nombre_canal, canal in (("gris", gris), ("v", valor_hsv)):
        umbral = _otsu_threshold(canal, valid_mask)

        for nombre_polaridad, prenda_clara in (
            ("prenda_clara", True),
            ("prenda_oscura", False),
        ):
            candidata = _mascara_desde_umbral(canal, umbral, valid_mask, prenda_clara)
            puntajes = _score_mask(candidata, valid_mask, borde_valido, area_valida_total)

            candidatas_info.append({
                "nombre": f"{nombre_canal}_{nombre_polaridad}",
                "umbral": umbral,
                "mask": candidata,
                **puntajes,
            })

    ganadora = max(candidatas_info, key=lambda c: c["puntaje_total"])
    mask_final = _componente_conexo_mas_grande(ganadora["mask"])

    if debug:
        return mask_final, candidatas_info

    return mask_final


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from preprocessing import preprocess_image

    if len(sys.argv) < 2:
        print("Uso: python segmentation.py <ruta_a_imagen>")
        sys.exit(0)

    imagen, mask_valida = preprocess_image(sys.argv[1])
    mask_prenda, candidatas = segment_garment(imagen, mask_valida, debug=True)

    print("Puntajes por candidata:")
    for c in sorted(candidatas, key=lambda c: -c["puntaje_total"]):
        print(
            f"  {c['nombre']:<20} umbral={c['umbral']:>3}  "
            f"area={c['area_proporcion']:.2%}  "
            f"compacidad={c['compacidad']:.3f}  "
            f"borde={c['ratio_contacto_borde']:.2%}  "
            f"puntaje_total={c['puntaje_total']:.3f}"
        )

    porcentaje_prenda = 100 * mask_prenda.sum() / mask_valida.sum()
    print(f"\nÁrea de la prenda segmentada: {porcentaje_prenda:.1f}% de la región válida")

    mask_visual = (mask_prenda.astype(np.uint8) * 255)
    cv2.imwrite("mascara_prenda.jpg", mask_visual)
    print("Máscara final guardada en mascara_prenda.jpg (para depurar)")
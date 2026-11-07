"""Parámetros del pipeline, agrupados en un solo lugar.

Cada dataclass es inmutable: para cambiar un valor se crea una copia
(`dataclasses.replace(cfg, lado_max=600)`), así una configuración nunca se modifica por accidente.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocesoConfig:
    # Tamaño: el lado mayor se reduce a `lado_max` px (nunca se amplía)
    lado_max: int = 800

    # Balance de blancos (mundo gris): corrige tintes de color de la luz.
    # `ganancia_max` limita cuánto puede cambiar cada canal (1.15 = máx. +-15 %).
    balance_blancos: bool = True
    ganancia_max: float = 1.15

    # Suavizado gaussiano para reducir ruido (0 = desactivado)
    sigma_suavizado: float = 1.0

    # Estiramiento de contraste por percentiles de la luminancia
    estirar_contraste: bool = True
    percentil_bajo: float = 1.0
    percentil_alto: float = 99.0
    expansion_max: float = 1.6  # tope del factor de expansión del contraste


@dataclass(frozen=True)
class DatasetConfig:
    """Paso 0: cómo se inventaría y se divide la base de imágenes."""

    fraccion_prueba: float = 0.30      # parte de la base que se reserva para evaluar al final
    semilla: str = "closet"            # mismo texto -> misma división (reproducible)
    umbral_casi_duplicado: int = 4     # distancia de Hamming (de 64 bits) para decir "casi la misma foto"
    umbral_firma_color: float = 5.0    # diferencia media de color (0-255) máxima para que dos fotos sean la misma
    mascaras_entrenamiento: int = 24   # fotos de entrenamiento cuya máscara se marca a mano
    mascaras_prueba: int = 12          # fotos de prueba cuya máscara se marca a mano


@dataclass(frozen=True)
class CalidadConfig:
    """Umbrales de las advertencias de calidad. Son orientativos: calíbralos con tus fotos."""

    lado_min: int = 300            # lado menor mínimo (px)
    nitidez_min: float = 20.0      # varianza del Laplaciano (menor = más borrosa)
    brillo_min: float = 60.0       # luminancia media 0-255
    contraste_min: float = 25.0    # desviación estándar de la luminancia
    frac_saturada_max: float = 0.40  # fracción de píxeles casi blancos (>= 250)


@dataclass(frozen=True)
class SegmentacionConfig:
    """Paso 2: segmentación con K-means. Las fracciones se miden sobre el lado mayor de la imagen."""

    k: int = 4                     # clusters de K-means
    semilla: int = 0               # semilla de K-means++ (mismo valor -> mismo resultado)
    peso_textura: float = 3.0      # peso de la textura local (0 = solo color)
    peso_espacial: float = 0.0     # peso de la posición (x, y); 0 = no se usa
    suavizado: float = 0.012       # radio del filtro de moda (borra moteado del fondo)
    nucleo: float = 0.004          # grosor mínimo de las regiones (corta puentes finos)
    cierre: float = 0.03           # radio del cierre morfológico (une rayas y estampados)
    ventana: float = 0.4           # ventana central donde se busca la prenda
    umbral_centro: float = 1.0     # un cluster es prenda si abunda al menos esto veces más en el centro (1.0 = no está sub-representado)

    # Paso 2b: refinar el borde con Otsu (apagado por defecto hasta medirlo con IoU sobre las máscaras de entrenamiento)
    refinar_otsu: bool = True
    banda_otsu: float = 0.010          # grosor de la banda alrededor del borde (fracción del lado mayor)
    separacion_otsu_min: float = 0.50  # separación mínima de Otsu (0-1) para fiarse de él; si no, se deja K-means
    cambio_max_otsu: float = 0.15      # el refinamiento no puede cambiar más de esta fracción del área


@dataclass(frozen=True)
class NormalizacionConfig:
    """Paso 3: dejar cada prenda recortada, enderezada y a un tamaño fijo."""

    lado: int = 256               # el lienzo final es lado x lado (la prenda queda centrada)
    margen: float = 0.06          # espacio libre alrededor de la prenda (fracción del lado del lienzo)
    enderezar: bool = True        # girar para poner el eje principal en vertical
    elongacion_min: float = 1.25  # solo se endereza si la prenda es claramente alargada (eje mayor / eje menor)
    giro_max: float = 45.0        # grados; un giro mayor se considera ambiguo y no se aplica
    giro_min: float = 3.0         # grados; giros más chicos no valen el remuestreo
    nucleo: float = 0.012         # radio de la erosión que da el "núcleo" para medir color (fracción del lado)


@dataclass(frozen=True)
class AlertasConfig:
    """Umbrales de las alertas de segmentación sospechosa. Son orientativos: se calibraron con solo 24 fotos."""

    area_max: float = 0.85         # la prenda ocupa casi toda la foto: probablemente se pegó fondo
    area_min: float = 0.08         # la prenda es diminuta: probablemente se perdió casi toda
    contraste_min: float = 25.0    # diferencia de color (Lab, ΔE) entre prenda y fondo; menor = se confunden
    fondo_sigma_max: float = 20.0  # variación de color del fondo (desv. est. en Lab); mayor = piso/tela estampada
    separacion: float = 0.03       # distancia (fracción del lado) que se deja entre la prenda y la zona de fondo medida


@dataclass(frozen=True)
class CaracteristicasConfig:
    """Paso 4: color, forma y textura de la prenda ya normalizada."""

    # Color (K-means sobre el núcleo de la prenda, en Lab)
    k_color: int = 3                  # colores que se buscan dentro de la prenda
    max_pixeles_color: int = 6000     # se submuestrea el núcleo para que K-means sea rápido
    fraccion_color_min: float = 0.12  # un color cuenta como "significativo" si cubre al menos esta fracción
    distancia_color_min: float = 18.0 # y si se distingue de los demás por al menos este ΔE (Lab)
    semilla: int = 0

    # Forma
    n_alturas: int = 10               # a cuántas alturas se mide la anchura de la prenda
    tramo_min: float = 0.03           # un tramo de prenda más angosto que esto (fracción del ancho) es ruido

    # Textura (Haralick sobre la luminancia del núcleo)
    niveles_gris: int = 16            # niveles de la matriz de co-ocurrencia (0-255 se reparte en partes iguales)
    distancia_haralick: int = 1       # separación entre píxeles para las 13 características
    distancia_patron: int = 4         # separación para medir rayas / cuadros (mira más lejos que un píxel)

    # Estampado: "liso" solo si TODO apunta a liso (alta precisión, recall moderado). Calibrado a ojo con las
    # 90 fotos de entrenamiento; no se tocó la prueba. Pliegues, acolchado y cuero salen como "con_textura".
    # Detalle sin sombras (pasa-banda en Lab): decide "liso" / "con_textura". Calibrado con 15 fotos normalizadas;
    # revisarlo con más fotos antes de fiarse del umbral.
    detalle_radio_chico: int = 1      # promedio local que quita el ruido del tejido
    detalle_radio_grande: int = 6     # promedio local que representa luz y pliegues suaves (se resta)
    detalle_umbral: float = 6.0       # ΔE (Lab) a partir del cual un píxel cuenta como detalle
    detalle_borde: int = 4            # px que se ignoran junto al borde del núcleo
    detalle_liso_max: float = 0.07    # fracción de detalle por debajo de la cual la prenda es lisa
    # (Umbrales anteriores, usados solo si falta `detalle_frac`)
    liso_homogeneidad_min: float = 0.84
    liso_contraste_max: float = 0.9
    liso_patron_max: float = 3.0      # contraste a `distancia_patron`, promedio de diagonales

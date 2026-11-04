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

"""
preprocessing.py
-----------------
Paso 2 del pipeline: Preprocesamiento de imágenes.

Recibe una imagen "cruda" (foto de una prenda) y la deja lista para la
etapa de segmentación: tamaño estándar, iluminación normalizada y ruido
reducido.

Criterio de "pureza" usado en este archivo:
- Se usa cv2/numpy SOLO como infraestructura (leer/guardar archivo,
  redimensionar, convertir espacio de color) porque esas operaciones
  no son parte de la teoría de procesamiento de imágenes que se evalúa.
- Los algoritmos que sí son contenido del curso (convolución/suavizado
  y ecualización de histograma) están implementados a mano con numpy,
  sin usar cv2.GaussianBlur, cv2.medianBlur ni cv2.createCLAHE.

Cada función es "pura": recibe una imagen (array de numpy) y devuelve
una imagen nueva, sin efectos secundarios. Esto facilita probarlas por
separado y encadenarlas en cualquier orden.
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Carga una imagen desde disco y la devuelve en formato RGB.

    Esto es infraestructura (leer bytes de un archivo), no un algoritmo
    de procesamiento de imágenes, por eso se usa cv2 directamente.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"No se pudo leer la imagen en: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Redimensiona la imagen a un tamaño estándar (ancho, alto),
    MANTENIENDO la proporción original y rellenando el sobrante
    con relleno negro (técnica de "letterbox").

    Antes esta función usaba cv2.resize directo a (512, 512), lo que
    ESTIRABA la imagen y distorsionaba la forma de la prenda cuando la
    foto original no era cuadrada (ej. fotos verticales de celular).
    Eso es un problema porque más adelante calcularemos características
    de forma (relación de aspecto, momentos de Hu) que dependen de que
    la prenda no esté deformada.

    Cómo funciona el letterbox:
    1. Calcula un factor de escala único que quepa en target_size sin
       deformar la imagen.
    2. Redimensiona con ese factor (mismo aspecto que el original).
    3. Centra el resultado sobre un lienzo negro de target_size.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def _gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Construye un kernel Gaussiano 2D de tamaño (size x size) a mano,
    usando directamente la fórmula de la distribución Gaussiana.

    G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))
    """
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / (2.0 * np.pi * sigma**2)
    return kernel / kernel.sum()  # normalizar para que no cambie el brillo global


def _convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolución 2D implementada a mano (sin scipy.signal ni cv2.filter2D).

    Aplica padding tipo "reflect" en los bordes y desliza el kernel
    sobre la imagen calculando la suma ponderada en cada posición.
    Es la operación fundamental detrás de cualquier filtro espacial
    (suavizado, realce de bordes, etc.).
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(channel, dtype=np.float64)

    h, w = channel.shape
    for i in range(h):
        for j in range(w):
            region = padded[i:i + k_h, j:j + k_w]
            out[i, j] = np.sum(region * kernel)

    return out


def denoise_image(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Reduce el ruido aplicando un filtro Gaussiano implementado a mano
    (kernel propio + convolución propia), canal por canal (R, G, B).

    Nota de rendimiento: la convolución con doble bucle es intencional
    para que el algoritmo sea explícito y trazable; para imágenes muy
    grandes es lento, por eso se recomienda trabajar con imágenes ya
    redimensionadas (ver resize_image) antes de este paso.
    """
    kernel = _gaussian_kernel(kernel_size, sigma)
    channels = cv2.split(image)  # separar R, G, B (esto sí es solo indexado, no algoritmo)

    blurred_channels = [_convolve2d(ch.astype(np.float64), kernel) for ch in channels]
    blurred = np.stack(blurred_channels, axis=-1)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _histogram_equalization_channel(channel: np.ndarray) -> np.ndarray:
    """
    Ecualización de histograma implementada a mano sobre un solo canal
    (valores 0-255), siguiendo la definición clásica:

    1. Calcular el histograma de niveles de gris.
    2. Calcular la función de distribución acumulada (CDF).
    3. Normalizar la CDF a 0-255 y usarla como tabla de mapeo (LUT).
    """
    histogram, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
    cdf = histogram.cumsum()

    cdf_min = cdf[cdf > 0].min()
    total_pixels = channel.size

    # Fórmula estándar de ecualización: escala la CDF a un rango 0-255
    cdf_normalized = np.round((cdf - cdf_min) / (total_pixels - cdf_min) * 255)
    cdf_normalized = np.clip(cdf_normalized, 0, 255).astype(np.uint8)

    return cdf_normalized[channel]  # aplicar la LUT a cada pixel


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    """
    Normaliza la iluminación mediante ecualización de histograma hecha
    a mano, aplicada solo al canal de luminosidad (Y de YCrCb) para no
    distorsionar los colores originales de la prenda.

    Se usa la conversión de espacio de color de cv2 (infraestructura),
    pero el algoritmo de ecualización en sí es propio.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    y_eq = _histogram_equalization_channel(y_channel)

    ycrcb_eq = cv2.merge((y_eq, cr_channel, cb_channel))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)


def prepare_for_display(path: str, target_size: tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Prepara una versión NÍTIDA de la imagen, solo para mostrarla en la
    app/armario digital (nunca se le aplica ecualización ni desenfoque).

    Solo hace lo mínimo de infraestructura: cargar y ajustar el tamaño
    manteniendo la proporción (letterbox), para que se vea consistente
    en la interfaz sin distorsionar ni desenfocar la prenda real.
    """
    image = load_image(path)
    return resize_image(image, target_size)


def preprocess_image(
    path: str,
    target_size: tuple[int, int] = (512, 512),
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Encadena todos los pasos de preprocesamiento en el orden correcto:
    cargar -> redimensionar -> normalizar iluminación -> reducir ruido.

    IMPORTANTE: el resultado de esta función es SOLO para uso interno
    del pipeline (segmentación, extracción de características). No se
    debe usar para mostrarle la prenda al usuario en el armario digital
    -- para eso está prepare_for_display(), que no desenfoca ni ecualiza.

    Se normaliza la iluminación antes de suavizar para que el filtro
    Gaussiano final también ayude a disimular el ruido que a veces
    introduce la ecualización de histograma en zonas muy uniformes.
    """
    image = load_image(path)
    image = resize_image(image, target_size)
    image = normalize_illumination(image)
    image = denoise_image(image, kernel_size, sigma)
    return image


if __name__ == "__main__":
    # Prueba rápida manual: reemplaza esta ruta por una imagen tuya en /data
    # y corre `python src/preprocessing.py` para verificar que no hay errores.
    import sys

    if len(sys.argv) < 2:
        print("Uso: python preprocessing.py <ruta_a_imagen>")
        sys.exit(0)

    # Versión para mostrar en la app (nítida, sin blur ni ecualización)
    display_result = prepare_for_display(sys.argv[1])
    display_bgr = cv2.cvtColor(display_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("resultado_para_mostrar.jpg", display_bgr)
    print("Versión para mostrar guardada en resultado_para_mostrar.jpg (nítida)")

    # Versión para el pipeline interno (ecualizada + suavizada)
    resultado = preprocess_image(sys.argv[1])
    print(f"Imagen preprocesada correctamente. Shape final: {resultado.shape}")
    salida = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
    cv2.imwrite("preprocesado_resultado.jpg", salida)
    print("Versión para el pipeline (interna) guardada en preprocesado_resultado.jpg")
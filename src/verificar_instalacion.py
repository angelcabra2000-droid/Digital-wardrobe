import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

# Imagen sintética: fondo gris con un rectángulo blanco (una "prenda" de mentira)
imagen = np.full((200, 300), 80, dtype=np.uint8)   # 200 alto x 300 ancho, gris oscuro
cv2.rectangle(imagen, (90, 40), (210, 160), 230, -1)  # rectángulo claro

print("Forma de la imagen:", imagen.shape)   # (alto, ancho)
print("Tipo de dato:", imagen.dtype)         # uint8 = enteros de 0 a 255

plt.imshow(imagen, cmap="gray")
plt.title("Instalación correcta")
plt.axis("off")
plt.show()
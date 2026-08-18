import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity as ssim
import cv2

# 1. Cargar las dos imágenes (deben tener el mismo tamaño)
# El paper asume que las imágenes se codifican en el espacio de color sRGB y no aplica transformaciones
img_referencia = cv2.imread('ds/ds_xray_1024/images_001/images/00000001_000.png', cv2.IMREAD_GRAYSCALE)
img_prueba = cv2.imread('ds/ds_xray_1024/images_001/images/00000001_001.png', cv2.IMREAD_GRAYSCALE)

# 2. Calcular el SSIM devolviendo el mapa de la imagen completa
# data_range=255 asume imágenes de 8-bits
score, ssim_map = ssim(img_referencia, img_prueba, data_range=255, full=True)

# 3. Crear el colormap personalizado descrito en el paper
# Nodos del gradiente (escala de -1.0 a 1.0 normalizada a 0.0 - 1.0 para matplotlib)
# -1.0 (0.0 en el nodo) -> Rojo
# < 0.0 (~0.499 en el nodo) -> Verde
#   0.0 (0.5 en el nodo) -> Negro
#   1.0 (1.0 en el nodo) -> Blanco
nodos = [0.0, 0.4999, 0.5, 1.0]
colores = [
    (1.0, 0.0, 0.0),  # Rojo (-1)
    (0.0, 1.0, 0.0),  # Verde (acercándose a 0 desde los negativos)
    (0.0, 0.0, 0.0),  # Negro (exactamente 0)
    (1.0, 1.0, 1.0)   # Blanco (1)
]

ssim_cmap = LinearSegmentedColormap.from_list("SSIM_paper_cmap", list(zip(nodos, colores)))

# 4. Mostrar el mapa de calor (Heatmap)
plt.figure(figsize=(8, 6))

# Asegurar que el rango vaya estrictamente de -1 a 1 para que el colormap se alinee correctamente
img_plot = plt.imshow(ssim_map, cmap=ssim_cmap, vmin=-1.0, vmax=1.0)

# Añadir la barra de color a la derecha
plt.colorbar(img_plot, label='Índice SSIM')
plt.title(f'Mapa de Calor SSIM (Score promedio: {score:.4f})')
plt.axis('off')

# Guardar la imagen o mostrarla
plt.savefig('heatmap_ssim.png', bbox_inches='tight')
plt.show()
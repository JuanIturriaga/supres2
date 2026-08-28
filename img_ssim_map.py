import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
from img import image_load, image_resize



def ssim_map_calculate(img_ref, img_test):
    """
    Calcula el mapa de similitud estructural (SSIM) entre dos imágenes y genera un mapa de calor.

    Args:
        img_ref (numpy.ndarray): Imagen de referencia.
        img_test (numpy.ndarray): Imagen de prueba.

    Returns:
        None
    """
    
    if img_ref.shape != img_test.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño y número de canales para calcular SSIM.")

    channel_axis = None
    if  img_ref.shape[2] == 3:
        channel_axis = 2  # Color image

    # 2. Calcular el SSIM devolviendo el mapa de la imagen completa
    # data_range=255 asume imágenes de 8-bits
    score, ssim_map = ssim(img_ref, img_test, channel_axis=channel_axis, data_range=255, full=True)
    # doc https://scikit-image.org/docs/stable/api/skimage.metrics.html
    
    if channel_axis is not None:
        # Si es una imagen a color, el mapa SSIM devuelto tiene 3 canales. Se promedia para obtener un solo mapa.
        ssim_map = np.mean(ssim_map, axis=channel_axis)
        
    return score, ssim_map


def img_ssim_map (img_ref, img_test, title=False, colorbar=False, save_to=None):
    """
    Calcula el mapa de similitud estructural (SSIM) entre dos imágenes y genera un mapa de calor.

    Args:
        img_ref (numpy.ndarray): Imagen de referencia.
        img_test (numpy.ndarray): Imagen de prueba.
        title (bool): Si es True, muestra el título con el score SSIM.
        colorbar (bool): Si es True, muestra la barra de color.
        save_to (str): Ruta para guardar la imagen del mapa de calor. Si es None, no se guarda.
    Returns:
        None
    """
    
    score, ssim_map = ssim_map_calculate(img_ref, img_test)
    
    # Crea el colormap personalizado descrito en el paper
    # Nodos del gradiente (escala de -1.0 a 1.0 normalizada a 0.0 - 1.0 para matplotlib)
    # -1.0 (0.0 en el nodo) -> Rojo
    # < 0.0 (~0.499 en el nodo) -> Verde
    #   0.0 (0.5 en el nodo) -> Negro
    #   1.0 (1.0 en el nodo) -> Blanco
    nodos = [0.0, 0.4999, 0.5, 1.0]
    # colores BGR (como los trabaja cv2)
    colores = [ 
        (0.0, 0.0, 1.0),  # Rojo (-1) 
        (0.0, 1.0, 0.0),  # Verde (acercándose a 0 desde los negativos)
        (0.0, 0.0, 0.0),  # Negro (exactamente 0)
        (1.0, 1.0, 1.0)   # Blanco (1)
    ]

    ssim_cmap = LinearSegmentedColormap.from_list("SSIM_paper_cmap", list(zip(nodos, colores)))
    
    ssim_map = (ssim_map + 1.0) / 2.0  # Normalizar de -1.0 - 1.0 a 0.0 - 1.0
    
    #crear una imagen combinando el mapa SSIM con el colormap
    heatmap_numpy = (ssim_cmap(ssim_map)[:, :, :3] * 255).astype(np.uint8)

    if save_to is not None:
        cv.imwrite(save_to, cv.cvtColor(heatmap_numpy, cv.COLOR_RGB2BGR))

    return score, heatmap_numpy, ssim_cmap


def plot_img_ssim (heat_map, score, ssim_cmap, vmin=-1, vmax=1):
    # Asegurar que el rango vaya estrictamente de -1 a 1 para que el colormap se alinee correctamente
    img_plot = plt.imshow(heat_map, cmap=ssim_cmap, vmin=vmin, vmax=vmax)

    # Añadir la barra de color a la derecha
    plt.colorbar(img_plot, label='Índice SSIM')
    plt.title(f'Mapa de Calor SSIM (Score promedio: {score:.4f})')
    plt.axis('off')

    # Guardar la imagen o mostrarla
    #plt.savefig('heatmap_ssim.png', bbox_inches='tight')
    plt.show()

    #guardar el plot en numpy
    plt.savefig('heatmap_ssim-con matplotlib.png', bbox_inches='tight')
    #heatmap_numpy = np.array(plt.gcf().canvas.renderer.buffer_rgba())
    
    


if __name__ == "__main__":

    # 1. Cargar las dos imágenes (deben tener el mismo tamaño)
    # El paper asume que las imágenes se codifican en el espacio de color sRGB y no aplica transformaciones
    img_referencia = image_load('ds/ds_xray_1024/images_001/images/00000001_000.png')
    img_prueba = image_load('ds/ds_xray_1024/images_001/images/00000001_001.png')

    #img_referencia = image_load('ds/random_shapes_64/00000000.png')
    #img_prueba = image_load('ds/random_shapes_64/00000000.png')

    #img_referencia = image_resize(img_referencia, 512, 512, interpolation='bicubic')
    #img_prueba = image_resize(img_prueba, 512, 512, interpolation='nearest')
    print (f'Image referencia shape: {img_referencia.shape}, Image prueba shape: {img_prueba.shape}')


    # Crea el colormap personalizado descrito en el paper
    # Nodos del gradiente (escala de -1.0 a 1.0 normalizada a 0.0 - 1.0 para matplotlib)
    # -1.0 (0.0 en el nodo) -> Rojo
    # < 0.0 (~0.499 en el nodo) -> Verde
    #   0.0 (0.5 en el nodo) -> Negro
    #   1.0 (1.0 en el nodo) -> Blanco
    nodos = [0.0, 0.4999, 0.5, 1.0]
    # colores RGB
    colores = [ 
        (1.0, 0.0, 0.0),  # Rojo (-1) 
        (0.0, 1.0, 0.0),  # Verde (acercándose a 0 desde los negativos)
        (0.0, 0.0, 0.0),  # Negro (exactamente 0)
        (1.0, 1.0, 1.0)   # Blanco (1)
    ]

    ssim_cmap_rgb = LinearSegmentedColormap.from_list("SSIM_paper_cmap", list(zip(nodos, colores)))


    score, heatmap, ssim_cmap = img_ssim_map(img_referencia, img_prueba, title=True, colorbar=True, save_to='heatmap_ssim_v2-concv.png')
    
    
    # BGR to rgb
    heatmap_rgb = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    heatmap_puro = (heatmap_rgb/255) * 2 -1
    
    plot_img_ssim (heatmap_rgb, score, ssim_cmap_rgb, -1, 1)
    
    
    #cv.imshow("Heatmap SSIM", heatmap)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    


def garbage():    
    

    
    channel_axis = None
    if  img_referencia.shape[2] == 3:
        channel_axis = 2  # Color image
        

    # 2. Calcular el SSIM devolviendo el mapa de la imagen completa
    # data_range=255 asume imágenes de 8-bits
    score, ssim_map = ssim(img_referencia, img_prueba, channel_axis=channel_axis, data_range=255, full=True)
    # doc https://scikit-image.org/docs/stable/api/skimage.metrics.html

    if channel_axis is not None:
        # Si es una imagen a color, el mapa SSIM devuelto tiene 3 canales. Se promedia para obtener un solo mapa.
        ssim_map = np.mean(ssim_map, axis=channel_axis)

    print (f"SSIM Score: {score:.4f}  MAX: {np.max(ssim_map):.4f}  MIN: {np.min(ssim_map):.4f}")
    print (f"SSIM ssim_map Shape: {ssim_map.shape}")  # Muestra la forma del mapa SSIM

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
    #plt.figure(figsize=(8, 6))

    # Asegurar que el rango vaya estrictamente de -1 a 1 para que el colormap se alinee correctamente
    img_plot = plt.imshow(ssim_map, cmap=ssim_cmap, vmin=-1.0, vmax=1.0)    

    # Añadir la barra de color a la derecha
    #plt.colorbar(img_plot, label='Índice SSIM')
    #plt.title(f'Mapa de Calor SSIM (Score promedio: {score:.4f})')
    plt.axis('off')

    # Guardar la imagen o mostrarla
    #plt.savefig('heatmap_ssim.png', bbox_inches='tight')
    #plt.show()

    #guardar el plot en numpy
    plt.savefig('heatmap_ssim-con matplotlib.png', bbox_inches='tight')
    #heatmap_numpy = np.array(plt.gcf().canvas.renderer.buffer_rgba())

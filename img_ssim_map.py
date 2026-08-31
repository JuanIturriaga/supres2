import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
from img import image_load, image_save, image_show, image_resize, images_load, images_save, images_resize



def ssim_map_calculate(img_ref, img_test):
    """
    Calcula el mapa de similitud estructural (SSIM) entre dos imágenes.

    Args:
        img_ref (numpy.ndarray): Imagen de referencia.
        img_test (numpy.ndarray): Imagen de prueba.

    Returns:
        score (float): Valor promedio del SSIM entre las dos imágenes.
        ssim_map (numpy.ndarray): Mapa de similitud estructural entre las dos imágenes.
            cada celda devuelve un valor entre -1 y 1, donde 1 indica una similitud perfecta.
        
        
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

def ssim_maps_calculate(images_ref, images_test):
    if len(images_ref) != len(images_test):
        raise ValueError("Las listas de imágenes deben tener el mismo tamaño.")

    scores = []
    ssim_maps = []
    for img_ref, img_test in zip(images_ref, images_test):
        score, ssim_map = ssim_map_calculate(img_ref, img_test)
        scores.append(score)
        ssim_maps.append(ssim_map)
        
    scores = np.array(scores)
    ssim_maps = np.array(ssim_maps)

    return scores, ssim_maps

def ssim_color_map (mode='rgb'):
    """
    Genera un colormap personalizado para el mapa SSIM basado en el paper.

    Args:
        mode (str, optional): Modo de color. puede ser 'rgb' o 'bgr'. Defaults to 'rgb'.

    Returns:
        LinearSegmentedColormap: Colormap personalizado para SSIM.
    """
    
    # Crea el colormap personalizado descrito en el paper
    # Nodos del gradiente (escala de -1.0 a 1.0 normalizada a 0.0 - 1.0 para matplotlib)
    # -1.0 (0.0 en el nodo) -> Rojo
    # < 0.0 (~0.499 en el nodo) -> Verde
    #   0.0 (0.5 en el nodo) -> Negro
    #   1.0 (1.0 en el nodo) -> Blanco
    nodos = [0.0, 0.4999, 0.5, 1.0]
    # colores BGR (como los trabaja cv2)
    
    colors_modes = {
        'bgr': [
            (0.0, 0.0, 1.0),  # Rojo (-1) 
            (0.0, 1.0, 0.0),  # Verde (acercándose a 0 desde los negativos)
            (0.0, 0.0, 0.0),  # Negro (exactamente 0)
            (1.0, 1.0, 1.0)   # Blanco (1)
        ],
        'rgb': [
            (1.0, 0.0, 0.0),  # Rojo (-1) 
            (0.0, 1.0, 0.0),  # Verde (acercándose a 0 desde los negativos)
            (0.0, 0.0, 0.0),  # Negro (exactamente 0)
            (1.0, 1.0, 1.0)   # Blanco (1)
        ]
    }
    
    colores = colors_modes[mode]    
    ssim_cmap = LinearSegmentedColormap.from_list(f"SSIM_cmap_{mode}", list(zip(nodos, colores)))
    
    return ssim_cmap

def image_ssim_map (ssim_map, save_to=None, mode='rgb'):
    """
    Genera un mapa de calor a partir de un mapa de similitud estructural (SSIM).

    Args:
        ssim_map (numpy.ndarray): Mapa de similitud estructural.
        save_to (str): Ruta para guardar la imagen del mapa de calor. Si es None, no se guarda.
        mode (str, optional): Modo de color. puede ser 'rgb' o 'bgr'. Defaults to 'rgb'.
    Returns:
        None
    """
    
    cmap = ssim_color_map(mode)
    
    ssim_map = (ssim_map + 1.0) / 2.0  # Normalizar de -1.0 - 1.0 a 0.0 - 1.0
    
    #crear una imagen combinando el mapa SSIM con el colormap
    heatmap_numpy = (cmap(ssim_map)[:, :, :3] * 255).astype(np.uint8)

    if save_to is not None:
        image_save(save_to, heatmap_numpy, mode=mode)
    
    return heatmap_numpy


def images_ssim_map(ssim_maps, save_to=None, mode='rgb'):
    """
    Genera mapas de calor a partir de una lista de mapas de similitud estructural (SSIM).

    Args:
        ssim_maps (numpy.ndarray): Lista de mapas de similitud estructural.
        save_to (str): Ruta para guardar las imágenes del mapa de calor. Si es None, no se guarda.
        mode (str, optional): Modo de color. puede ser 'rgb' o 'bgr'. Defaults to 'rgb'.
    Returns:
        numpy.ndarray: Lista de mapas de calor generados a partir de los mapas SSIM.
    """

    cmap = ssim_color_map(mode)
    
    heatmaps = []
    for i, ssim_map in enumerate(ssim_maps):
        #crear una imagen combinando el mapa SSIM con el colormap
        ssim_map = (ssim_map + 1.0) / 2.0  # Normalizar de -1.0 - 1.0 a 0.0 - 1.0
        heatmap_numpy = (cmap(ssim_map)[:, :, :3] * 255).astype(np.uint8)
            
        if save_to is not None:
            image_save(f"{save_to}_{i}.png", heatmap_numpy, mode=mode)
        
        heatmaps.append(heatmap_numpy)

    return np.array(heatmaps)


def plot_ssim_map (ssim_map, score):
    # Asegurar que el rango vaya estrictamente de -1 a 1 para que el colormap se alinee correctamente
    
    cmap = ssim_color_map('rgb')
    img_plot = plt.imshow(ssim_map, cmap=cmap, vmin=-1, vmax=1)

    # Añadir la barra de color a la derecha
    plt.colorbar(img_plot, label='Índice SSIM')
    plt.title(f'Mapa de Calor SSIM (Score promedio: {score:.4f})')
    plt.axis('off')

    # Guardar la imagen o mostrarla
    plt.show()


def _test_arrays():
    images_originales = images_load('ds/random_shapes_64/', mode='rgb', extensions=['.png'], recursive=False, max_images=3)
    
    images_r = images_resize(images_originales, 512, 512, interpolation='bicubic')
    images_t = images_resize(images_originales, 512, 512, interpolation='nearest')
    
    scores, ssim_maps = ssim_maps_calculate(images_r, images_t)
    
    images_m = images_ssim_map(ssim_maps, save_to=None, mode='rgb')
    
    for i in range(scores.shape[0]):
        print(f"Imagen {i}: SSIM Score: {scores[i]:.4f} shape: {ssim_maps[i].shape} MAX: {np.max(ssim_maps[i]):.4f}  MIN: {np.min(ssim_maps[i]):.4f}")
    
    images_save('heatmaps_ssim', images_m, mode='rgb', prefix='heatmap_', start_index=0)
    images_save('images_referencia', images_r, mode='rgb', prefix='ref_', start_index=0)
    images_save('images_prueba', images_t, mode='rgb', prefix='test_', start_index=0)
    
    
    
def _test_single():
    img_original = image_load('ds/random_shapes_64/00000000.png', mode='rgb')
    
    img_referencia = image_resize(img_original, 512, 512, interpolation='bicubic')
    img_prueba = image_resize(img_original, 512, 512, interpolation='nearest')
    
    score, ssim_map = ssim_map_calculate(img_referencia, img_prueba)
    
    print(f"SSIM Score: {score:.4f} shape: {ssim_map.shape} MAX: {np.max(ssim_map):.4f}  MIN: {np.min(ssim_map):.4f}")
            
    heatmap = image_ssim_map(ssim_map, save_to='heatmap_ssim.png', mode='rgb')
    
    image_show(heatmap)


if __name__ == "__main__":
    
    _test_arrays()
    
    
    
def garbage2():

    # 1. Cargar las dos imágenes (deben tener el mismo tamaño)
    # El paper asume que las imágenes se codifican en el espacio de color sRGB y no aplica transformaciones
    img_referencia = image_load('ds/ds_xray_1024/images_001/images/00000001_000.png')
    #img_prueba = image_load('ds/ds_xray_1024/images_001/images/00000001_001.png')
    #invertir imagen de referencia
    img_prueba = cv.bitwise_not(img_referencia)
    
    
    
    

    #img_referencia = image_load('ds/random_shapes_64/00000000.png')
    #img_prueba = image_load('ds/random_shapes_64/00000000.png')
    #img_referencia = image_resize(img_referencia, 512, 512, interpolation='bicubic')
    #img_prueba = image_resize(img_prueba, 512, 512, interpolation='nearest')
        
    #print (f'Image referencia shape: {img_referencia.shape}, Image prueba shape: {img_prueba.shape}')

    #calcula el mapa ssim y el score
    #score, ssim_map = ssim_map_calculate(img_referencia, img_prueba)
    
    #print(f"SSIM Score: {score:.4f} shape: {ssim_map.shape} MAX: {np.max(ssim_map):.4f}  MIN: {np.min(ssim_map):.4f}")
            
    #heatmap = img_ssim_map(ssim_map, save_to='heatmap_ssim_v2-concv.png', mode='rgb')
    #image_show(heatmap)
    
    #plot_ssim_map(ssim_map, score)
    
    
    
    


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

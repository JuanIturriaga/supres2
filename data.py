'''
Carga o crea data sets en base a parámetros indicados en un diccionario de configuración.

{
    "dataset_path": "ds/xray500",
    "dataset_gen": "random_textures",
    "dataset_count": 500
}

'''

import numpy as np
from img import images_load
from img_gen import img_random_textures, img_random_shapes

def dataset_load(params, verbose=False):
    path = params.get('dataset_path', None)
    max_images = params.get('dataset_count', None)
    
    if max_images < 0:
        max_images = None  # Si se pasa un valor negativo, se cargan todas las imágenes
        
    if path is None:
        raise ValueError("El parámetro 'dataset_path' es obligatorio.")
    
    images = images_load(path, max_images=max_images, verbose=verbose)
    return images

def dataset_random_textures(params, verbose=False):
    count = params.get('dataset_count', 100)
    
    shape = params.get('dataset_shape', None)
    if shape is None:
        output_size = params.get('output_size', 128)
        shape = (output_size, output_size, 3)
    
    
    texture_count = params.get('dataset_texture_count', 10)
    
    images = []
    for i in range(count):
        img = img_random_textures(shape=shape, texture_count=texture_count)
        images.append(img)
        if verbose and (i + 1) % 100 == 0:
            print(f"Generadas {i + 1}/{count} imágenes de texturas aleatorias")
    
    if verbose:
        print(f"Generadas {len(images)} imágenes de texturas aleatorias")
    
    return np.array(images)

def dataset_random_shapes(params, verbose=False):
    count = params.get('dataset_count', 100)
    
    shape = params.get('dataset_shape', None)
    if shape is None:
        output_size = params.get('output_size', 128)
        shape = (output_size, output_size, 3)
    
    shape_count = params.get('dataset_shape_count', 10)
    
    images = []
    for i in range(count):
        img = img_random_shapes(shape=shape, shape_count=shape_count)
        images.append(img)
        if verbose and (i + 1) % 100 == 0:
            print(f"Generadas {i + 1}/{count} imágenes de formas aleatorias")
            
    if verbose:
        print(f"Generadas {len(images)} imágenes de formas aleatorias")

    return np.array(images)

dataset_fuctions = {
    'random_textures': dataset_random_textures,
    'random_shapes': dataset_random_shapes,
    'load': dataset_load
}

def dataset_factory (params, verbose=False):
    '''
    Carga o crea un dataset en base a los parámetros indicados en el diccionario de configuración.
    
    Parámetros:
        params (dict): diccionario de configuración con los siguientes campos:
            - dataset_path (str): ruta del dataset a cargar o crear.
            - dataset_gen (str): tipo de dataset a crear (opcional).
            - dataset_count (int): cantidad de imágenes a crear (opcional).
    
    Devuelve:
        images (numpy): lista de imágenes cargadas o creadas.
    '''
    
    dataset_gen = params.get('dataset_gen', 'load')
    dataset_func = dataset_fuctions.get(dataset_gen, None)
    
    if dataset_func is None:
        raise ValueError(f"Función de dataset no encontrada para el gen: {dataset_gen}")
    
    images = dataset_func(params, verbose=verbose)
    
    if verbose:
        print(f"Dataset cargado/creado con formato {images[0].shape} y {len(images)} imágenes.")
    
    return images


if __name__ == "__main__":
    
    # Ejemplo de uso
    params = {
        "dataset_path": "./ds/ds_xray_1024/images_001/images",
        "dataset_gen": "random_shapes",
        "dataset_count": 3,
        "output_size": 512,
        "dataset_texture_count": 4,
        "dataset_shape_count": 40
    }
    
    images = dataset_factory(params, verbose=True)
    print(f"Dataset cargado/creado con {len(images)} imágenes.")
    
    #mostrar todas las imágenes cargadas
    import cv2
    for i, img in enumerate(images):
        cv2.imshow(f"Imagen {i+1}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
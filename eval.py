# Funciones para evaluar distancias de imágnes y métricas de similitud entre imágenes.

import numpy as np
from skimage.metrics import structural_similarity as mssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse

def eval_mssim(orig, test):
    channels = orig.shape[-1] if len(orig.shape) == 3 else 1
    if channels == 1:
        return mssim(np.squeeze(orig), np.squeeze(test))
    else:
        ssim_values = []
        for i in range(channels):
            ssim_values.append(mssim(orig[..., i], test[..., i]))
        return np.mean(ssim_values)

def eval_psnr(orig, test):
    result = float('inf')
    err = eval_mse(orig, test)
    if err != 0:
        result = psnr(orig, test, data_range=255)
    return result

def eval_r2(orig, test):
    return r2(orig.flatten(), test.flatten())

def eval_mse(orig, test):
    return mse(orig.flatten(), test.flatten())

def eval_rmse(orig, test):
    return np.sqrt(mse(orig.flatten(), test.flatten()))

eval_functions = {
    'mssim': eval_mssim,
    'psnr': eval_psnr,
    'r2': eval_r2,
    'mse': eval_mse,
    'rmse': eval_rmse
}

def evaluate_metrics(original_images, test_images, metrics_list=['mssim', 'psnr', 'r2', 'mse', 'rmse']):
    ''' 
    Evaluar métricas entre dos listas de imágenes originales y de prueba.
    
    Parámetros:
        original_images (arreglo numpy de imágenes con 1 o 3 canales): lista de imágenes originales.
        test_images (arreglo numpy de imágenes con 1 o 3 canales): lista de imágenes de prueba, las imágenes deben tener el mismo tamaño que las originales.
        metrics_list (lista de str, opcional): lista de métricas a evaluar (por defecto ['mssim', 'psnr', 'r2', 'mse', 'rmse']).
    
    Devuelve:
        metrics (lista de listas): lista de listas con los valores de las métricas para cada par de imágenes.
            el orden de los valores de las métricas es el mismo que el de metrics_list.
    '''
        
    metrics = []    
    for orig, test in zip(original_images, test_images):
        image_metrics = []
        for metric in metrics_list:
            eval_func = eval_functions.get(metric)
            if eval_func is not None:
                image_metrics.append(eval_func(orig, test))
            else:
                image_metrics.append(None)  # Si la métrica no es válida, agregar None
        metrics.append(image_metrics)
    
    return metrics


# run python eval.py for testing
from img import images_load, images_resize
from img_gen import img_random_shapes
import pandas as pd
import numpy as np


if __name__ == "__main__":
    
    # Define the path to the folder containing the images and the number of images to load
    path = ".\\ds\\ds_xray_1024"
    count = 100
    print (f"Loading {count} images from {path}")    
        
    #Define el tamáño de las imágenes y el factor de reducción
    size = (64, 64)  # Tamaño de prueba para redimensionar las imágenes
    factor = 2  # Factor de reducción para la imagen de prueba        
    print (f"Image Size: {size}, Reduction Factor: {factor}")
    shape = (size[0], size[1], 3)
    
    # Carga las imágenes originales de una carpeta específica
    original_image = images_load(path, mode='rgb', recursive=True, max_images=count)
    # original_image = img_random_shapes(shape)
    
    ratio = 0.8
    y_train = original_image[:int(len(original_image) * ratio)]
    y_val = original_image[int(len(original_image) * ratio):]
        
    # Redimensiona las imágenes originales y las de prueba según el factor de reducción
    original_image_resized = images_resize(y_val, size[0], size[1], interpolation='bicubic')
    test_image_resized = images_resize(original_image_resized, size[0]//factor, size[1]//factor, interpolation='bicubic')
    
    # Imprime la forma de la primer imagen de cada arreglo
    print (f"Resized Original Image Shape: {original_image_resized[0].shape}")
    print (f"Resized Test Image Shape: {test_image_resized[0].shape}")
        
    # Define las interpolaciones (que se van a comparar) y métricas a evaluar
    interpolations = ['bicubic', 'bilinear', 'nearest']
    #metrics_list=['mssim', 'psnr', 'mse', 'rmse', 'r2']
    metrics_list=['mssim', 'psnr', 'mse']
    
    # Crea dataframe pandas con las metricas y sus valores
    metrics_df = pd.DataFrame(columns=metrics_list)

    # Evalúa las métricas para cada interpolación y almacena los resultados medios de cada métrica en el dataframe
    for i in interpolations:
        print(f"Evaluating metrics for interpolation: {i}")
        #redimensiona las imágenes de prueba a su tamaño original usando la interpolación actual
        tests = images_resize(test_image_resized, size[0], size[1], interpolation=i) 
        print(f"Finalizado redimensionamiento: cantidad de imágenes {len(test_image_resized)} forma: {tests[0].shape}")
        #evalua todas las métricas para las imágenes originales y las de prueba redimensionadas
        results = evaluate_metrics(original_image_resized, tests, metrics_list=metrics_list)   
        print(f"Finalizado evaluación de métricas: metrics={metrics_list}")
        #almacena los resultados medios de cada métrica en el dataframe      
        metrics_df.loc[i] = np.mean(results, axis=0)        
        print(f"Finalizado almacenamiento de resultados en dataframe.")
    
    #print pandas dataframe
    print(metrics_df)
# Funciones para evaluar distancias de imágnes y métricas de similitud entre imágenes.

import numpy as np
from skimage.metrics import structural_similarity as mssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse


def evaluate_metrics(original_images, test_images, metrics_list=['mssim', 'psnr', 'r2', 'mse', 'rmse']):
        
    metrics = []    
    for orig, test in zip(original_images, test_images):
        image_metrics = []
        for metric in metrics_list:
            if metric == 'mssim':
                # conciderar imágenes en escala de grises o color 
                if orig.ndim == 2 or orig.shape[2] == 1:
                    image_metrics.append(mssim(np.squeeze(orig), np.squeeze(test)))
                else:
                    # ver como funciona ssim en imágenes a color !!!!!! 
                    image_metrics.append(mssim(np.squeeze(orig), np.squeeze(test), channel_axis=-1))
            elif metric == 'psnr':
                image_metrics.append(psnr(orig, test))
            elif metric == 'r2':
                image_metrics.append(r2(orig.flatten(), test.flatten()))
            elif metric == 'mse':
                image_metrics.append(mse(orig.flatten(), test.flatten()))
            elif metric == 'rmse':
                image_metrics.append(np.sqrt(mse(orig.flatten(), test.flatten())))
        metrics.append(image_metrics)
    
    return metrics


from img import load_image, resize_image

if __name__ == "__main__":
    # Cargar imágenes originales y de prueba
    original_image = load_image(".\\ds\\ds_xray_1024\\images_001\\images\\00000001_000.png", mode='grayscale')
    test_image = load_image(".\\ds\\ds_xray_1024\\images_001\\images\\00000001_001.png", mode='grayscale')
    
    size = (254, 254)  # Tamaño de prueba para redimensionar las imágenes
    factor = 2  # Factor de reducción para la imagen de prueba
    
    print (f"Original Image Shape: {original_image.shape}")
    print (f"Test Image Shape: {test_image.shape}")
    
    #normalizar imágenes a rango [0, 1]
    #original_image = original_image / 255.0
    #test_image = test_image / 255.0
    
    # Redimensionar imágenes si es necesario
    original_image_resized = resize_image(original_image, size[0], size[1], interpolation='bicubic')
    test_image_resized = resize_image(test_image, size[0]//factor, size[1]//factor, interpolation='bicubic')
    test_image_bicubic = resize_image(test_image_resized, size[0], size[1], interpolation='bicubic')
    test_image_bilineal = resize_image(test_image_resized, size[0], size[1], interpolation='bilinear')
    test_image_nearest = resize_image(test_image_resized, size[0], size[1], interpolation='nearest')
    
    # Evaluar métricas
    metrics_list=['mssim', 'psnr', 'mse', 'rmse', 'r2']
    metrics_bicubic = evaluate_metrics([original_image_resized], [test_image_bicubic], metrics_list=metrics_list)
    metrics_bilineal = evaluate_metrics([original_image_resized], [test_image_bilineal], metrics_list=metrics_list)
    metrics_nearest = evaluate_metrics([original_image_resized], [test_image_nearest], metrics_list=metrics_list)
    
    #crear dataframe pandas con las metricas y sus valores
    import pandas as pd
    cols = ['interpol'] + metrics_list
    
    metrics_df = pd.DataFrame(columns=metrics_list)
    metrics_df.loc['bicubic'] = metrics_bicubic[0]
    metrics_df.loc['bilineal'] = metrics_bilineal[0]
    metrics_df.loc['nearest'] = metrics_nearest[0]
    
    
    
    #print pandas dataframe
    print(metrics_df)
    
    
    


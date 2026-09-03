"""
Función para correr experimentos de super resolución según los parámetros de entrada
Los paso básicos son: 
    PASO 1: Preparación del dataset (cargar o crear dataset)
    PASO 2: Creación del modelo (crear o cargar modelo)
    PASO 3: Entrenamiento del modelo (entrenar el modelo con los datos de entrenamiento y validación)
    PASO 4: Evaluación del modelo (evaluar el modelo con los datos de prueba y calcular métricas)
"""
import os

from data import dataset_factory
from models import models_factory
from train import DatasetIterator, train_supres_model
from eval import evaluate_metrics
from img import images_resize, images_save
from img_ssim_map import ssim_maps_calculate, images_ssim_map, plot_ssim_map
import numpy as np
import os
import csv
import subprocess
import tensorflow as tf
from datetime import datetime

def experiment_setup (params, verbose=False):
    """Escanea la carpeta de resultados buscando un nuevo id para el experimento

    Args:
        params (diccionario): debe incluir los paráemtros del experimento
        verbose (bool, optional): Imprimir por consola resultados. Defaults to False.
    
    Returns:
        str: experiment_id (str)
    """
       
    output_folder = params.get('experiment_output_folder', './results')
    tag = params.get('experiment_tag', 'supres')
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Creando carpeta de resultados: {output_folder}")
    
    num = 0
    flag = True
    while flag:
        experiment_id = f"{tag}_{num:06d}"
        experiment_folder = os.path.join(output_folder, experiment_id)
        if not os.path.exists(experiment_folder):
            flag = False
            os.makedirs(experiment_folder)
            print(f"ID EXPERIMENTO: {experiment_id}")
            print(f"Creando carpeta de resultados del experimento: {experiment_folder}")
        else:
            num += 1                        
            
    return experiment_id

def experiment_resize(inputs_set, width, height, method='bilinear'):
    """
    Redimensiona un conjunto de imágenes a un tamaño específico utilizando un método de interpolación dado.
    !!!! Pensado para agregar más métodos de redimencionamiento en el futuro, 
    !!!! por ejemplo uso de un modelo de super resolución preentrenado para redimensionar las imágenes.

    Args:
        inputs_set (numpy array): Conjunto de imágenes a redimensionar.
        width (int): Ancho deseado para las imágenes redimensionadas.
        height (int): Alto deseado para las imágenes redimensionadas.
        method (str, optional): Método de interpolación a utilizar. Puede ser 'bilinear', 'nearest', 'bicubic', 'area' o 'lanc'. Defaults to 'bilinear'.

    Returns:
        numpy array: Conjunto de imágenes redimensionadas.
    """
        
    resized_set = images_resize(inputs_set, width, height, interpolation=method)
    return resized_set

def metrics_to_csv(results, metrics_list, filename, path= './'):
    """
    Guardar resultados de métricas en un archivo csv

    Args:
        results (lista de listas): Resultados de las métricas para cada imagen. (una imágen por fila, cada métrica en una columna)
        metrics_list (lista de str): Nombres de las métricas.
        filename (str): Nombre del archivo CSV donde se guardarán los resultados.
        path (str, optional): Carpeta donde se guardará el archivo CSV. Defaults to './'.
    """    
    with open(os.path.join(path, filename), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_index'] + metrics_list)
        for i, result in enumerate(results):
            writer.writerow([i] + result)
            
def gpu_info():
    """
    Obtiene información sobre la GPU disponible en el sistema.

    tf.config.experimental.get_device_details() no expone memoria total/libre,
    por lo que se consulta nvidia-smi; si no está disponible se usa
    tf.config.experimental.get_memory_info() (sólo memoria en uso por TF).

    Returns:
        str: Información sobre la GPU, incluyendo nombre, memoria total y memoria libre.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        return "No GPU available."

    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            encoding='utf-8', timeout=5
        )
        gpu_info = []
        for line in smi_output.strip().splitlines():
            name, mem_total, mem_used, mem_free = [v.strip() for v in line.split(',')]
            gpu_info.append(
                f"GPU: {name}, Memory Total: {float(mem_total):.2f} MB, "
                f"Memory Used: {float(mem_used):.2f} MB, Memory Free: {float(mem_free):.2f} MB"
            )
        return "\n".join(gpu_info)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        gpu_info = []
        for i, gpu in enumerate(gpus):
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get('device_name', 'Unknown GPU')
            try:
                mem = tf.config.experimental.get_memory_info(f'GPU:{i}')
                current_mb = mem.get('current', 0) / (1024 ** 2)
                peak_mb = mem.get('peak', 0) / (1024 ** 2)
                gpu_info.append(f"GPU: {name}, Memory In Use: {current_mb:.2f} MB, Peak: {peak_mb:.2f} MB")
            except Exception:
                gpu_info.append(f"GPU: {name}, Memory info unavailable (nvidia-smi not found).")
        return "\n".join(gpu_info)

def experiment_supres_basico (params, verbose=False):
    """
    Función para correr un experimento de super resolución según los parámetros de entrada.
    
    Args:
        params (dict): Diccionario de parámetros del experimento.
        verbose (bool): Si es True, imprime información adicional durante la ejecución.
    
    Returns:
        dict: Diccionario con los resultados del experimento, incluyendo métricas de evaluación.
    """
    
    # PASO 0: Preparación del experimento (crear carpeta de resultados y asignar un ID único)
    start_time = datetime.now()    
    params['experiment_start_time'] = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        
    exp_id = experiment_setup(params, verbose=verbose)
    exp_result_path = os.path.join(params.get('experiment_output_folder', './results'), exp_id) 
    
    params['experiment_id'] = exp_id
    params['experiment_output_folder'] = exp_result_path
    
    #obtener datos de la placa de video 
    params['gpu_info'] = gpu_info()
    
    
    # PASO 1: Preparación del dataset
    start_time_dataset = datetime.now()
    if verbose:
        print(f"PASO 1 ({start_time_dataset.strftime('%Y-%m-%d %H:%M:%S.%f')}): Preparación del dataset...")
    dataset = dataset_factory(params, verbose=verbose)
    end_time_dataset = datetime.now()
    params['dataset_time'] = (end_time_dataset - start_time_dataset).total_seconds()
    
    # PASO 2: Creación del modelo
    start_time_model = datetime.now()
    if verbose:
        print(f"PASO 2 ({start_time_model.strftime('%Y-%m-%d %H:%M:%S.%f')}): Creación del modelo...")
    model = models_factory(params, verbose=verbose)
    
    # PASO 3: Entrenamiento del modelo
    start_time_train = datetime.now()
    if verbose:
        print(f"PASO 3 ({start_time_train.strftime('%Y-%m-%d %H:%M:%S.%f')}): Entrenamiento del modelo...")
    history, x_train, y_train, x_val, y_val = train_supres_model(params, model, dataset, verbose=verbose)
    end_time_train = datetime.now()
    params['train_time'] = (end_time_train - start_time_train).total_seconds()    
    
    # guardar el historial de entrenamiento en un archivo CSV
    history_file = os.path.join(exp_result_path, f'training_history_{exp_id}.csv')
    with open(history_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch'] + list(history.history.keys()))
        for i in range(len(history.history['loss'])):
            writer.writerow([i] + [history.history[key][i] for key in history.history.keys()])
    
    # guardar el modelo entrenado
    model_file = os.path.join(exp_result_path, f'trained_model_{exp_id}.keras')
    model.save(model_file)
    
    # PASO 4: Evaluación del modelo
    if verbose:
        print("PASO 4: Evaluación del modelo...")
        
    exp_img_path = os.path.join(exp_result_path, f'images_{exp_id}')
    os.makedirs(exp_img_path, exist_ok=True)
        
    # metricas (TODO: debería ser un parámetro de entrada)
    metrics_list=['mssim', 'psnr', 'mse']
        
    # Predicciones del modelo sobre el conjunto de validación
    tf.keras.backend.clear_session()
    #cargar el modelo entrenado
    model = tf.keras.models.load_model(model_file, compile=False)
    predict_gen = DatasetIterator(x_val, batch_size=params.get('predict_batch_size', params.get('train_batch_size', 32)))
    y_pred = model.predict(predict_gen)
    
    # Las referencias y entradas permanecen en uint8; sólo la salida del modelo se desnormaliza.
    originals_set = y_val
    inputs_set = x_val
    predictions_set = (y_pred * 255).astype(np.uint8)
    
    # Cálculo de métricas comparando orginales y predicciones del modelo
    predict_metrics = evaluate_metrics(originals_set, predictions_set, metrics_list=metrics_list)
    metrics_to_csv(predict_metrics, metrics_list=metrics_list, filename='metrics_predict.csv', path=exp_result_path) 
    predict_means = {metric: np.mean([result[i] for result in predict_metrics]) for i, metric in enumerate(metrics_list)}
    predict_variances = {metric: np.var([result[i] for result in predict_metrics]) for i, metric in enumerate(metrics_list)}
    
    # Armar un dataframe con media y varianza
    # La fila es cada método y las columnas son las métricas (media y varianza)
    results_summary_header = ['method'] + [f"{metric}_mean" for metric in metrics_list] + [f"{metric}_var" for metric in metrics_list]
    results_summary = [['predict'] + [predict_means[metric] for metric in metrics_list] + [predict_variances[metric] for metric in metrics_list]]
    scores, ssim_maps = ssim_maps_calculate(originals_set[:3], predictions_set[:3])
    predict_ssim_map_rgb = images_ssim_map(ssim_maps, save_to=None, mode='rgb')
    images_save(os.path.join(exp_img_path,'originals'), originals_set[:3], mode='rgb')
    images_save(os.path.join(exp_img_path,'predicts'), predictions_set[:3], mode='rgb')
    images_save(os.path.join(exp_img_path,'predicts_ssim_maps'), predict_ssim_map_rgb, mode='rgb')
    
    # Comparativas: se calculan las métricas del original vs el redimensionado con diferentes métodos de interpolación.
    shape = originals_set[0].shape
    original_width, original_height = shape[1], shape[0] #TODO: revisar orden
    print(f"Original Image Shape: {shape} (Width: {original_width}, Height: {original_height})")    

    interpolation_methods = ['bicubic', 'bilinear', 'nearest']
    for method in interpolation_methods:
        
        if verbose:
            print(f"Evaluando método de comparación: {method}...")
        
        resized_set = experiment_resize(inputs_set, original_height, original_width, method=method)
        
        # Calcular métricas para cada método de redimencionamiento
        method_metric = evaluate_metrics(originals_set, resized_set, metrics_list=metrics_list)
        
        # Guardar results en csv con el indice de la imágen en la primera columna
        metrics_to_csv(method_metric, metrics_list=metrics_list, filename=f'metrics_{method}.csv', path=exp_result_path)
        
        # Calcular la media y varianza de cada métrica para cada método
        method_means = {metric: np.mean([result[i] for result in method_metric]) for i, metric in enumerate(metrics_list)}
        method_variances = {metric: np.var([result[i] for result in method_metric]) for i, metric in enumerate(metrics_list)}
        results_summary.append([method] + [method_means[metric] for metric in metrics_list] + [method_variances[metric] for metric in metrics_list])
        
        # Calcular mapas de similitud estructural (SSIM) para cada método y guardarlos en la carpeta de resultados
        scores, ssim_maps = ssim_maps_calculate(originals_set[:3], resized_set[:3])
        method_ssim_map_rgb = images_ssim_map(ssim_maps, save_to=None, mode='rgb')
        images_save(os.path.join(exp_img_path, method), resized_set[:3], mode='rgb')
        images_save(os.path.join(exp_img_path, f'{method}_ssim_map'), method_ssim_map_rgb, mode='rgb')
        
        
    # Guardar el resumen de resultados en un archivo CSV
    with open(os.path.join(exp_result_path, 'results_summary.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(results_summary_header)
        for row in results_summary:
            writer.writerow(row) 
            
    return {
        'experiment_id': exp_id,
        'results_summary': results_summary,
        'results_summary_header': results_summary_header,
        'predict_metrics': predict_metrics,
        'predict_means': predict_means,
        'predict_variances': predict_variances
    }                 
    
import json        
    
if __name__ == "__main__":
    # Ejemplo de uso de la función experiment_supres_basico
    params = {
        'dataset_type': 'random_shapes',
        'dataset_path': './ds/ds_xray_1024',
        'dataset_count': 20000,
        'model_architecture': 'conv0',
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'loss_function': 'mse',
        'train_shuffle': False,
        'train_batch_size': 32,
        'train_epochs': 20,
        'train_ratio': 0.8,
        'input_size': 128,
        'input_channels': 3,
        'output_size': 256,
        'input_interpolation_method': 'bicubic', 
        'experiment_tag': 'sp_random_shapes',
        'experiment_id': 'sp_000000',
        'experiment_type': 'experiment_supres_basico',
        'experiment_output_folder': './results',
        'experiment_description': 'Experimento de super resolución con modelo conv0 y dataset formas aleatorias'        
    }
    
    results = experiment_supres_basico(params, verbose=True)
    
    experiment_folder = params.get('experiment_output_folder', './results')
        
    #save params as json 
    with open(os.path.join(experiment_folder, 'params.json'), 'w') as f:
        json.dump(params, f)
    
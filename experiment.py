"""
Función para correr experimentos de super resolución según los parámetros de entrada
Los paso básicos son: 
    PASO 1: Preparación del dataset (cargar o crear dataset)
    PASO 2: Creación del modelo (crear o cargar modelo)
    PASO 3: Entrenamiento del modelo (entrenar el modelo con los datos de entrenamiento y validación)
    PASO 4: Evaluación del modelo (evaluar el modelo con los datos de prueba y calcular métricas)
"""
from data import dataset_factory
from models import models_factory
from train import train_supres_model
from eval import evaluate_metrics
from img import images_resize
import numpy as np

def experiment_supres_basico (params, verbose=False):
    """
    Función para correr un experimento de super resolución según los parámetros de entrada.
    
    Args:
        params (dict): Diccionario de parámetros del experimento.
        verbose (bool): Si es True, imprime información adicional durante la ejecución.
    
    Returns:
        dict: Diccionario con los resultados del experimento, incluyendo métricas de evaluación.
    """
    
    # PASO 1: Preparación del dataset
    if verbose:
        print("PASO 1: Preparación del dataset...")
    dataset = dataset_factory(params, verbose=verbose)
    
    # PASO 2: Creación del modelo
    if verbose:
        print("PASO 2: Creación del modelo...")
    model = models_factory(params, verbose=verbose)
    
    # PASO 3: Entrenamiento del modelo
    if verbose:
        print("PASO 3: Entrenamiento del modelo...")
    history, x_train, y_train, x_val, y_val = train_supres_model(params, model, dataset, verbose=verbose)
        
    # Predicciones del modelo sobre el conjunto de validación
    y_pred = model.predict(x_val)
    
    # Desnormalizar las imágenes de salida y de validación si es necesario
    originals_set = (y_val * 255).astype(np.uint8)  # desnormalizar las imágenes de validación
    inputs_set = (x_val * 255).astype(np.uint8)
    predictions_set = (y_pred * 255).astype(np.uint8)
    
    # Comparativas: se toma las imágenes de entrada y se las agranda utilizando técnicas tradicionales
    shape = originals_set[0].shape
    bilineal_set = images_resize(inputs_set, shape[0], shape[1], interpolation='bilinear')
    bicubic_set = images_resize(inputs_set, shape[0], shape[1], interpolation='bicubic')
    nearest_set = images_resize(inputs_set, shape[0], shape[1], interpolation='nearest')
    
    # PASO 4: Evaluación del modelo
    if verbose:
        print("PASO 4: Evaluación del modelo...")
        
    metrics_list = ['mssim', 'psnr', 'mse']
    predict_results = evaluate_metrics(originals_set, predictions_set, metrics_list=metrics_list)
    bilinear_results = evaluate_metrics(originals_set, bilineal_set, metrics_list=metrics_list)
    bicubic_results = evaluate_metrics(originals_set, bicubic_set, metrics_list=metrics_list)
    nearest_results = evaluate_metrics(originals_set, nearest_set, metrics_list=metrics_list)
    
    predict_means = {metric: np.mean([result[i] for result in predict_results]) for i, metric in enumerate(metrics_list)}
    bilinear_means = {metric: np.mean([result[i] for result in bilinear_results]) for i, metric in enumerate(metrics_list)}
    bicubic_means = {metric: np.mean([result[i] for result in bicubic_results]) for i, metric in enumerate(metrics_list)}
    nearest_means = {metric: np.mean([result[i] for result in nearest_results]) for i, metric in enumerate(metrics_list)}
    
    # print means 
    if verbose:
        print("Resultados de evaluación:")
        print(f"Predicciones del modelo: {predict_means}")
        print(f"Agradado bilineal: {bilinear_means}")
        print(f"Agradado bicúbico: {bicubic_means}")
        print(f"Agradado vecino más cercano: {nearest_means}")
        
    return {
        'history': history,
        'predict_results': predict_results,
        'bilinear_results': bilinear_results,
        'bicubic_results': bicubic_results,
        'nearest_results': nearest_results
    }
    
import json        
    
if __name__ == "__main__":
    # Ejemplo de uso de la función experiment_supres_basico
    params = {
        'dataset_type': 'load',
        'dataset_path': './ds/ds_xray_1024',
        'dataset_count': 10000,
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
        'input_interpolation_method': 'bicubic'
    }
    
    results = experiment_supres_basico(params, verbose=True)
    
    #save params as json 
    with open('params.json', 'w') as f:
        json.dump(params, f)
    
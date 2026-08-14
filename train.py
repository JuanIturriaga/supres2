"""
    _summary_
    Training script for the model.
    
    La idea es condensar distintas funciones de entrenamiento en el script    


"""
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from img import images_resize, images_blur

# AUX: clase iteradora para evitar problema de memoria
class DatasetIterator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        # Calcula cuántos lotes hay por época
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Extrae exactamente <batch_size> imágenes de la RAM y las prepara
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def generate_input(params, dataset_output):
    """
    Genera un conjunto de datos de entrada a partir del conjunto de datos original.
    
    Args:
        params: Diccionario de parámetros que contiene la configuración para generar los datos de entrada.
        dataset_output: Conjunto de datos original (imágenes tamaño completo).
        
    Returns:
        input_dataset: Conjunto de datos de entrada generado a partir del conjunto original.
    """    
    
    input_size = params.get('input_size', 14)
    output_size = params.get('output_size', 28)
    
    #No se utiliza la GPU por problemas de memoria
    with tf.device('/CPU:0'):
        if input_size != output_size:
            method = params.get('input_interpolation_method', 'bicubic')
            #dataset_input = tf.image.resize(y_train, (input_size, input_size), method=method).numpy()
            dataset_input = images_resize (dataset_output, input_size, input_size, interpolation=method)            
            
        else:
            #hacer un blur gassiano para simular baja resolución
            method = params.get('input_blur_kernel_size', 5)
            #dataset_input = tf.nn.avg_pool2d(dataset_output, ksize=2, strides=1, padding='SAME').numpy()
            dataset_input = images_blur(dataset_output, kernel_size=method)

    return dataset_input
  

def train_supres_model(params, model, dataset, verbose=False):
    """
    Entrena el modelo con los datos de entrenamiento y validación proporcionados.
    
    Args:
        model: Modelo de Keras de supres a entrenar.
        dataset: Conjunto de datos completo (imágenes tamaño completo). Se dividirá en entrenamiento y validación según el ratio especificado en params.
        params: Diccionario con parámetros de entrenamiento (batch_size, epochs, etc.).
        verbose: Booleano para imprimir información adicional durante el entrenamiento.
    
    Returns:
        history: Objeto History que contiene información sobre el entrenamiento.
    """
    
    # Baraja el dataset con imágenes de alta resolución
    shuffle = params.get('train_shuffle', False)
    if shuffle:
        np.random.shuffle(dataset)
        
    # Divide el dataset en entrenamiento y validación
    ratio = params.get('train_ratio', 0.8)
    y_train = dataset[:int(len(dataset) * ratio)]
    y_val = dataset[int(len(dataset) * ratio):]
    
    # Genera los datos de entrada a partir de las imágenes de alta resolución
    # Reduce el tamaño al tamaño del input o si el input es igual al output lo blurea para simular baja resolución
    x_train = generate_input(params, y_train)
    x_val = generate_input(params, y_val)
    
    # Normaliza los datos de entrada y salida a [0, 1] para el entrenamiento
    x_train = x_train / 255.0
    y_train = y_train / 255.0
    x_val = x_val / 255.0
    y_val = y_val / 255.0    
    
    # Configura los parámetros de entrenamiento
    batch_size = params.get('train_batch_size', 32)
    epochs = params.get('train_epochs', 10)

    # Crea los iteradores para no subir todo a la memoria de la GPU
    train_gen = DatasetIterator(x_train, y_train, batch_size)
    val_gen = DatasetIterator(x_val, y_val, batch_size)
    
    # Entrena el modelo
    if verbose:
        print(f"Entrenando el modelo con {len(x_train)} imágenes de entrenamiento y {len(x_val)} imágenes de validación...")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")        
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2
    )
    
    return history, x_train, y_train, x_val, y_val
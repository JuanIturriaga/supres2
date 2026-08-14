import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model

"""
Crea un modelos según la arquitectura especificada en un diccionario de parámetros.
Además de crear el modelo, también compila el modelo con el optimizador y la función de pérdida especificados en los parámetros.
Los parámetros se pasan en un diccionario, por ejemplo:

{
    "model_file_": "Si se especifica, se carga el modelo desde este archivo y se ignoran los parámetros de arquitectura.",
    "model_architecture_": "Si se especifica, se crea un nuevo modelo con esta arquitectura. Por ejemplo, 'autoencoder_64to128' o 'autoencoder_32to64'.",
    "model_architecture": "conv0",
    "input_size": 64,
    "input_channels": 3,
    "optimizer": "adam",
    "learning_rate": 0.0001,
    "loss_function": "MSE",
    "output_size": 128,
}

"""

def create_sr_conv0 (params):
    """
    Crea un modelo autoencoder para superresolución de imágenes
    el modelo es convolucional simple, con upsampling directo en la primera capa

    Args:
        input_shape_low_res (tuple): Dimensiones de la imagen de entrada de baja resolución (altura, ancho, canales).
        
    Returns:
        tensorflow.keras.models.Model: El modelo autoencoder de superresolución.
    """
    
    input_shape_low_res = params.get('model_input_shape', (14, 14, 1))
    
    input_img = Input(shape=input_shape_low_res)
    
    # upsamplig 1: 14x14 -> 28x28
    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(input_img) # 28x28x128
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x) # 28x28x64
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x) # 28x28x32
    x = Conv2D(16, (9, 9), activation='relu', padding='same')(x) # 28x28x16
    # Capa de Salida
    num_output_channels = input_shape_low_res[-1] # obtiene la cantidad de canales de salida según la entrada.
    decoded = Conv2D(num_output_channels, (3, 3), activation='sigmoid', padding='same')(x) # 28x28x1

    # Creación del modelo
    autoencoder = Model(input_img, decoded)
    return autoencoder


def create_sr_conv2 (params):
    """
    Crea un modelo autoencoder para superresolución de imágenes
    el modelo es convolucional simple, sin upsampling, el input es igual al output
    
    Modelo basado en SRCNN de Doung et al. (2016) "Image Super-Resolution Using Deep Convolutional Networks"

    Args:
        params (dict): Diccionario de parámetros del modelo, debe contener la clave 'input_shape_low_res' con las dimensiones de la imagen de entrada de baja resolución (altura, ancho, canales).
        
    Returns:
        tensorflow.keras.models.Model: El modelo autoencoder de superresolución.
    """
    
    input_shape_low_res = params.get('model_input_shape', (14, 14, 1))
    
    input_img = Input(shape=input_shape_low_res)
    
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(input_img) 
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)     
    # Capa de Salida
    num_output_channels = input_shape_low_res[-1] # obtiene la cantidad de canales de salida según la entrada.
    decoded = Conv2D(num_output_channels, (5, 5), activation='sigmoid', padding='same')(x) # 28x28x1

    # Creación del modelo
    autoencoder = Model(input_img, decoded)
    return autoencoder


def optimizer_factory (optimizer_name, learning_rate):
    """
    Crea un optimizador de Keras según el nombre y la tasa de aprendizaje especificados.

    Args:
        optimizer_name (str): Nombre del optimizador ('adam', 'sgd', 'rmsprop', etc.).
        learning_rate (float): Tasa de aprendizaje para el optimizador.

    Returns:
        tensorflow.keras.optimizers.Optimizer: Instancia del optimizador.
    """
    if optimizer_name.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimización no soportada: {optimizer_name}")

def load_model_from_file (params):
    """
    Carga un modelo de Keras desde un archivo especificado en los parámetros.

    Args:
        params (dict): Diccionario de parámetros del modelo, debe contener la clave 'model_file' con la ruta del archivo del modelo.

    Returns:
        tensorflow.keras.models.Model: El modelo cargado desde el archivo.
    """
    model_file = params.get('model_file', None)
    if model_file is None:
        raise ValueError("Se requiere 'model_file' en los parámetros para cargar un modelo desde un archivo.")
    
    return tf.keras.models.load_model(model_file)


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
   
    
def loss_function_factory (loss_name):
    """
    Crea una función de pérdida de Keras según el nombre especificado.

    Args:
        loss_name (str): Nombre de la función de pérdida ('mse', 'mae', 'binary_crossentropy', etc.).

    Returns:
        tensorflow.keras.losses.Loss: Instancia de la función de pérdida de Keras.
    """
    if loss_name.lower() == 'mse':
        return tf.keras.losses.MeanSquaredError() 
    elif loss_name.lower() == 'mae':
        return tf.keras.losses.MeanAbsoluteError()
    elif loss_name.lower() == 'ssim':
        return SSIMLoss
    else:
        raise ValueError(f"Función de pérdida no soportada: {loss_name}")
    
model_architectures = {
    'conv0': create_sr_conv0,
    'conv2': create_sr_conv2,
    'load': load_model_from_file
}

def models_factory (params, verbose=False):    
    
    model_input_size = params.get('input_size', 14)
    model_input_channels = params.get('input_channels', 3)
    model_input_shape = (model_input_size, model_input_size, model_input_channels)
    params['model_input_shape'] = model_input_shape    
    
    model_architecture = params.get('model_architecture', 'conv0')
    model_func = model_architectures.get(model_architecture, None)
    if model_func is None:
        raise ValueError(f"Arquitectura de modelo no soportada: {model_architecture}")

    model = model_func(params)
    
    optimizer_name = params.get('optimizer', 'adam')
    learning_rate = params.get('learning_rate', 0.001)  
    
    optimizer = optimizer_factory(optimizer_name, learning_rate)
    
    loss_function_name = params.get('loss_function', 'mse')
    loss_function = loss_function_factory(loss_function_name)
    
    model.compile(optimizer=optimizer, loss=loss_function)
    
    if verbose:
        print(f"Modelo creado con arquitectura '{model_architecture}', optimizador '{optimizer_name}', tasa de aprendizaje {learning_rate}, función de pérdida '{loss_function_name}'.")
        print(f"Resumen del modelo:")
        model.summary()    
    
    return model
    
    
if __name__ == "__main__":
    # Ejemplo de uso
    params = {
        "model_architecture": "conv2",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "loss_function": "mse",
        "model_input_size": 256,
        "model_input_channels": 1
    }
    
    model = models_factory(params, verbose=True)
    
    
from img_gen import img_random_shapes
import os
import cv2
from data import dataset_random_shapes


if __name__ == "__main__":
    # Define the path to the folder containing the images and the number of images to load
    path = ".\\ds\\random_shapes_64"
    count = 100
    
    ds = dataset_random_shapes(params={'dataset_count': count, 'output_size': 64, 'input_channels': 3}, verbose=True)
    
    # validar si existe la carpeta, si no existe crearla
    if not os.path.exists(path):   
        os.makedirs(path)
        nro = 0
    else:
        # encontrar el nro mas alto de imágenes con formato 00000000.png
        existing_files = [f for f in os.listdir(path) if f.endswith('.png')]
        existing_numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
        nro = max(existing_numbers) + 1 if existing_numbers else 0
        
    for i, img in enumerate(ds):
        filename = os.path.join(path, f"{nro + i:08d}.png")
        cv2.imwrite(filename, img)
        

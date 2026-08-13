import cv2 as cv
import numpy as np
import os

def image_load(path, mode='color'):
    #cargar una imagen desde un archivo
    map = {
        'color': cv.IMREAD_COLOR,
        'grayscale': cv.IMREAD_GRAYSCALE,
        'unchanged': cv.IMREAD_UNCHANGED
    }
    return cv.imread(path, map.get(mode, cv.IMREAD_COLOR))

def images_load (folder_path, mode='color', extensions=['.png', '.jpg', '.jpeg'], recursive=False, max_images=None, dim = (-1,-1), verbose=False):
    #cargar todas las imagenes de una carpeta
    
    images = []
    if recursive:
        for root, _, files in os.walk(folder_path):
            for filename in files:
                if any(filename.endswith(ext) for ext in extensions):
                    img = image_load(os.path.join(root, filename), mode)
                    if dim != (-1,-1):
                        img = image_resize(img, dim[0], dim[1], interpolation='bicubic')
                    if img is not None:
                        images.append(img)                        
                        if max_images is not None and len(images) >= max_images:
                            break
            if max_images is not None and len(images) >= max_images:
                break
    else:
        for filename in os.listdir(folder_path):
            if any(filename.endswith(ext) for ext in extensions):
                img = image_load(os.path.join(folder_path, filename), mode)
                if dim != (-1,-1):
                    img = image_resize(img, dim[0], dim[1], interpolation='bicubic')
                if img is not None:
                    images.append(img)
                    if max_images is not None and len(images) >= max_images:
                        break
    if verbose:
        print(f"Cargadas {len(images)} imágenes desde {folder_path}")
        
    return np.array(images)

def image_resize(image, width, height, interpolation='bilinear'):
    #redimensionar una imagen a un tamaño específico
    map = {
        'bilinear': cv.INTER_LINEAR,
        'nearest': cv.INTER_NEAREST,
        'bicubic': cv.INTER_CUBIC,
        'area': cv.INTER_AREA,
        'lanc': cv.INTER_LANCZOS4
    }
    interpolation_value = map.get(interpolation)    
    if interpolation_value is None:
        raise ValueError(f"Invalid interpolation method: {interpolation}")
    return cv.resize(image, (width, height), interpolation=interpolation_value)

def image_blur(image, kernel_size=5):
    #aplicar un blur gaussiano a una imagen
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

def images_resize(images, width, height, interpolation='bilinear'):
    #redimensionar una lista de imagenes a un tamaño específico
    return [image_resize(image, width, height, interpolation) for image in images]

if __name__ == "__main__":
    image = image_load(".\\ds\\ds_xray_1024\\images_001\\images\\00000001_000.png")
    print(f"Original Image Shape: {image.shape}")
    resized_image = image_resize(image, 200, 200, interpolation='bicubic')
    cv.imshow("Resized Image", resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    


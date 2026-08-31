import cv2 as cv
import numpy as np
import os

def image_load(path, mode='rgb'):
    #cargar una imagen desde un archivo
    map = {
        'rgb': cv.IMREAD_COLOR,
        'bgr': cv.IMREAD_COLOR,
        'grayscale': cv.IMREAD_GRAYSCALE,
        'unchanged': cv.IMREAD_UNCHANGED
    }
    img = cv.imread(path, map.get(mode,  map.get(mode)))
    if mode == 'rgb' and img is not None:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def image_save(path, image, mode='rgb'):
    #guardar una imagen en un archivo
    if mode == 'rgb':
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(path, image)
    
def images_load (folder_path, mode='rgb', extensions=['.png', '.jpg', '.jpeg'], recursive=False, max_images=None, dim = (-1,-1), verbose=False):
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

def images_save (folder_path, images, mode='rgb', prefix='image_', start_index=0):
    #guardar una lista de imagenes (numpy array) en una carpeta
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, img in enumerate(images):
        image_save(os.path.join(folder_path, f"{prefix}{start_index + i:06d}.png"), img, mode)

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

def images_resize(images, width, height, interpolation='bilinear'):
    #redimensionar una lista de imagenes a un tamaño específico
    return np.array([image_resize(image, width, height, interpolation) for image in images])

def image_blur(image, kernel_size=5):
    #aplicar un blur gaussiano a una imagen
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

def images_blur(images, kernel_size=5):
    #aplicar un blur gaussiano a una lista de imagenes
    return np.array([image_blur(image, kernel_size) for image in images])

def image_map_compare (img_orginal, img_test, method='diff'):
    #compara dos imagenes y devuelve una imagen con la diferencia
    if method == 'diff':
        return cv.absdiff(img_orginal, img_test)
    elif method == 'sub':
        return cv.subtract(img_orginal, img_test)
    elif method == 'add':
        return cv.add(img_orginal, img_test)
    else:
        raise ValueError(f"Invalid comparison method: {method}")
    
    
def image_show (image, title="Image", color='rgb', wait_key=0):
    #mostrar una imagen en una ventana
    
    if color == 'rgb':
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    cv.imshow(title, image)
    cv.waitKey(wait_key)
    cv.destroyAllWindows()

if __name__ == "__main__":
    #image = image_load(".\\ds\\ds_xray_1024\\images_001\\images\\00000001_000.png")
    image = image_load(".\\ds\\random_shapes_64\\00000000.png", mode='grayscale')
    if image is not None:
        print(f"Original Image Shape: {image.shape}")
        resized_image = image_resize(image, 256, 256, interpolation='bicubic')
        image_show(resized_image, title="Resized Image", color='grayscale', wait_key=0)
    else:
        print("Error: No se pudo cargar la imagen.")

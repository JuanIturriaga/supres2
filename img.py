import cv2 as cv

def load_image(path, mode='color'):
    #cargar una imagen desde un archivo
    map = {
        'color': cv.IMREAD_COLOR,
        'grayscale': cv.IMREAD_GRAYSCALE,
        'unchanged': cv.IMREAD_UNCHANGED
    }
    return cv.imread(path, map.get(mode, cv.IMREAD_COLOR))

def resize_image(image, width, height, interpolation='bilinear'):
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

if __name__ == "__main__":
    image = load_image(".\\ds\\ds_xray_1024\\images_001\\images\\00000001_000.png")
    print(f"Original Image Shape: {image.shape}")
    resized_image = resize_image(image, 200, 200, interpolation='bicubic')
    cv.imshow("Resized Image", resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    


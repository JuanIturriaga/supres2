import cv2
import numpy as np
import random


# crear una imagen con formas aleatorias
def img_random_shapes (shape=(128,128,3), shape_count=10):
    """
    Genera una imagen PNG de tamaño NxN con figuras geométricas aleatorias.
    
    Parámetros:
    - shape: Tupla con las dimensiones de la imagen (alto, ancho, canales).
    - shape_count: Número de figuras a dibujar en la imagen.
    
    Returns: 
    - Imagen generada como un array de NumPy.
    
    """
    
    # Crear un lienzo de NxN píxeles con 3 canales (RGB)
    # dtype=np.uint8 asegura que los valores estén en el rango 0-255
    imagen = np.zeros(shape, dtype=np.uint8)
    
    # Llenar el fondo de color blanco (opcional, si prefieres fondo negro, comenta esta línea)
    imagen.fill(255)

    for _ in range(shape_count):
        # Elegir un tipo de figura al azar
        tipo_figura = random.choice(['linea', 'rectangulo', 'circulo'])
        
        # Generar un color aleatorio (B, G, R en OpenCV)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Grosor aleatorio: -1 rellena la figura (solo para círculos y rectángulos)
        grosor = random.choice([-1, 1, 2, 4, 6])

        if tipo_figura == 'linea':
            # Puntos aleatorios para la línea
            pt1 = (random.randint(0, shape[1]), random.randint(0, shape[0]))
            pt2 = (random.randint(0, shape[1]), random.randint(0, shape[0]))
            # Las líneas no aceptan grosor -1, forzamos un valor positivo
            grosor_linea = random.randint(1, 6)
            cv2.line(imagen, pt1, pt2, color, grosor_linea)
            
        elif tipo_figura == 'rectangulo':
            # Vértices opuestos aleatorios
            pt1 = (random.randint(0, shape[1]), random.randint(0, shape[0]))
            pt2 = (random.randint(0, shape[1]), random.randint(0, shape[0]))
            cv2.rectangle(imagen, pt1, pt2, color, grosor)
            
        elif tipo_figura == 'circulo':
            # Centro y radio aleatorios
            centro = (random.randint(0, shape[1]), random.randint(0, shape[0]))
            radio = random.randint(10, min(shape[0], shape[1]) // 3) # Radio limitado para que entre en la imagen
            cv2.circle(imagen, centro, radio, color, grosor)
    
    return imagen


def img_random_textures (shape=(128,128,3), texture_count=5):
    """
    Genera una imagen PNG de tamaño NxN con texturas aleatorias.
    
    Parámetros:
    - shape: Tupla con las dimensiones de la imagen (alto, ancho, canales).
    - texture_count: Número de texturas a aplicar en la imagen.
    
    Returns: 
    - Imagen generada como un array de NumPy.
    """
    
    # Crear un lienzo de NxN píxeles con 3 canales (RGB)
    imagen = np.zeros(shape, dtype=np.uint8)
    
    # Color de fondo aleatorio
    if shape[2] == 1:
        color_fondo = random.randint(50, 200)
        imagen.fill(color_fondo)
    else:
        color_fondo = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        imagen[:] = color_fondo

    for _ in range(texture_count):
        # Elegir un tipo de textura al azar
        tipo_textura = random.choice([
            'lineas_horizontales', 'lineas_verticales', 'lineas_diagonales',
            'puntos', 'gradiente', 'rejilla', 'ondas', 'ruido_estructurado'
        ])
        
        # Generar colores aleatorios
        if shape[2] == 1:
            color = random.randint(0, 255)
        else:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Crear una máscara aleatoria para aplicar la textura solo en ciertas áreas
        mascara = np.random.random((shape[0], shape[1])) > 0.3  # 70% de la imagen tendrá textura
        
        if tipo_textura == 'lineas_horizontales':
            espaciado = random.randint(3, 15)
            grosor = random.randint(1, 3)
            for y in range(0, shape[0], espaciado):
                if y + grosor < shape[0]:
                    imagen[y:y+grosor, :] = np.where(mascara[y:y+grosor, :][..., np.newaxis], 
                                                   color, imagen[y:y+grosor, :])
        
        elif tipo_textura == 'lineas_verticales':
            espaciado = random.randint(3, 15)
            grosor = random.randint(1, 3)
            for x in range(0, shape[1], espaciado):
                if x + grosor < shape[1]:
                    imagen[:, x:x+grosor] = np.where(mascara[:, x:x+grosor][..., np.newaxis], 
                                                   color, imagen[:, x:x+grosor])
        
        elif tipo_textura == 'lineas_diagonales':
            espaciado = random.randint(5, 20)
            for i in range(-shape[1], shape[1], espaciado):
                for y in range(shape[0]):
                    x = y + i
                    if 0 <= x < shape[1] and mascara[y, x]:
                        cv2.line(imagen, (x, y), (x, y), color, 1)
        
        elif tipo_textura == 'puntos':
            densidad = random.randint(50, 200)  # Número de puntos
            radio = random.randint(1, 4)
            for _ in range(densidad):
                x, y = random.randint(0, shape[1]-1), random.randint(0, shape[0]-1)
                if mascara[y, x]:
                    cv2.circle(imagen, (x, y), radio, color, -1)
        
        elif tipo_textura == 'gradiente':
            # Gradiente horizontal o vertical
            direccion = random.choice(['horizontal', 'vertical'])
            if direccion == 'horizontal':
                for x in range(shape[1]):
                    intensidad = int(255 * (x / shape[1]))
                    if shape[2] == 1:
                        color_grad = min(255, max(0, color + intensidad - 128))
                    else:
                        color_grad = (
                            min(255, max(0, color[0] + intensidad - 128)),
                            min(255, max(0, color[1] + intensidad - 128)),
                            min(255, max(0, color[2] + intensidad - 128))
                        )
                    imagen[:, x] = np.where(mascara[:, x][..., np.newaxis], 
                                          color_grad, imagen[:, x])
            else:  # vertical
                for y in range(shape[0]):
                    intensidad = int(255 * (y / shape[0]))
                    if shape[2] == 1:
                        color_grad = min(255, max(0, color + intensidad - 128))
                    else:
                        color_grad = (
                            min(255, max(0, color[0] + intensidad - 128)),
                            min(255, max(0, color[1] + intensidad - 128)),
                            min(255, max(0, color[2] + intensidad - 128))
                        )
                    imagen[y, :] = np.where(mascara[y, :][..., np.newaxis], 
                                          color_grad, imagen[y, :])
        
        elif tipo_textura == 'rejilla':
            espaciado = random.randint(8, 25)
            grosor = random.randint(1, 2)
            # Líneas horizontales
            for y in range(0, shape[1], espaciado):
                if y + grosor < shape[1]:
                    imagen[y:y+grosor, :] = np.where(mascara[y:y+grosor, :][..., np.newaxis], 
                                                   color, imagen[y:y+grosor, :])
            # Líneas verticales
            for x in range(0, shape[0], espaciado):
                if x + grosor < shape[0]:
                    imagen[:, x:x+grosor] = np.where(mascara[:, x:x+grosor][..., np.newaxis], 
                                                   color, imagen[:, x:x+grosor])
        
        elif tipo_textura == 'ondas':
            # Ondas sinusoidales
            frecuencia = random.uniform(0.1, 0.5)
            amplitud = random.randint(10, 30)
            direccion = random.choice(['horizontal', 'vertical'])
            
            for i in range(shape[1] if direccion == 'horizontal' else shape[0]):
                if direccion == 'horizontal':
                    desplazamiento = int(amplitud * np.sin(2 * np.pi * frecuencia * i))
                    for j in range(shape[0]):
                        y_onda = j + desplazamiento
                        if 0 <= y_onda < shape[0] and mascara[y_onda, i]:
                            imagen[y_onda, i] = color
                else:  # vertical
                    desplazamiento = int(amplitud * np.sin(2 * np.pi * frecuencia * i))
                    for j in range(shape[1]):
                        x_onda = j + desplazamiento
                        if 0 <= x_onda < shape[1] and mascara[i, x_onda]:
                            imagen[i, x_onda] = color
        
        elif tipo_textura == 'ruido_estructurado':
            # Ruido con patrones
            tamano_bloque = random.randint(2, 8)
            for y in range(0, shape[0], tamano_bloque):
                for x in range(0, shape[1], tamano_bloque):
                    if random.random() > 0.5 and y < shape[0] and x < shape[1]:
                        y_fin = min(y + tamano_bloque, shape[0])
                        x_fin = min(x + tamano_bloque, shape[1])
                        
                        # Aplicar solo donde la máscara lo permite
                        bloque_mascara = mascara[y:y_fin, x:x_fin]
                        imagen[y:y_fin, x:x_fin] = np.where(bloque_mascara[..., np.newaxis], 
                                                          color, imagen[y:y_fin, x:x_fin])
    
    return imagen
    
    
    
    
if __name__ == "__main__":
    
    # Generar una imagen con formas aleatorias
    img_shapes = img_random_shapes(shape=(480, 640, 3), shape_count=15)
    cv2.imshow("Imagen con Formas Aleatorias", img_shapes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Generar una imagen con texturas aleatorias
    img_textures = img_random_textures(shape=(480, 640, 3), texture_count=7)
    cv2.imshow("Imagen con Texturas Aleatorias", img_textures)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

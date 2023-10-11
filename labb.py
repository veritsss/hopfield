import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('triangulo3.jpg', cv2.IMREAD_GRAYSCALE)  # Cambia 'imagen.jpg' por la ruta de tu imagen

# Verificar si la imagen es más pequeña que 4x4
if image is not None and (image.shape[0] >= 4 and image.shape[1] >= 4):
    # Redimensionar la imagen a 4x4 utilizando interpolación
    image = cv2.resize(image, (100,100), interpolation=cv2.INTER_LINEAR)

    # Binarizar la imagen
    _, binary_image = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)

    # Convertir la imagen binaria en un patrón unidimensional
    binary_pattern = binary_image.flatten()

    # Separar los valores por comas
    binary_pattern_str = ', '.join(map(str, binary_pattern))
    binary_pattern_str = '[' + binary_pattern_str + ']'
    # Mostrar la imagen original
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    plt.show()

    # Mostrar el patrón binario
    print("\nPatrón Binario:")
    print(binary_pattern_str)
else:
    print("La imagen es demasiado pequeña para redimensionar a 4x4 o no se pudo cargar la imagen.")
# Crear una cadena de caracteres con los corchetes alrededor del patrón binario
binary_pattern_str = ''.join(map(str, binary_pattern))

# Guardar la cadena en un archivo de texto
with open('triangulo3.txt', 'w') as file:
    file.write(binary_pattern_str)


    

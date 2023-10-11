import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self):
        self.weights = None

    def train(self, patterns):
        # Determinar el tamaño de los patrones y configurar la matriz de pesos en consecuencia
        pattern_size = patterns[0].shape[0]
        self.weights = np.zeros((pattern_size, pattern_size))

        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)  # Convierte el patrón en un vector columna
            self.weights += np.dot(pattern, pattern.T)
            np.fill_diagonal(self.weights, 0)  # La diagonal debe ser cero

    def energy(self, state):
        return -0.5 * np.dot(np.dot(state.T, self.weights), state)

    def update(self, state, max_iterations=1000):
        for _ in range(max_iterations):
            new_state = np.sign(np.dot(self.weights, state))
            if np.array_equal(new_state, state):
                break
            state = new_state
        return state;

# Función para cargar patrones desde archivos de texto
def load_patterns_from_files(pattern_files):
    patterns = []
    for file in pattern_files:
        with open(file, 'r') as f:
            pattern_str = f.read()
            pattern = np.array([int(bit) for bit in pattern_str])
            patterns.append(pattern)
    return patterns

# Lista de archivos de patrones
pattern_files = ["triangulo.txt","triangulo.txt","triangulo.txt","triangulo.txt","triangulo2.txt","triangulo2.txt","triangulo2.txt","triangulo2.txt","triangulo2.txt"]  # Agrega los nombres de tus archivos

# Cargar patrones desde archivos de texto
patterns = load_patterns_from_files(pattern_files)

# Crear la red Hopfield y entrenarla
hopfield_net = HopfieldNetwork()
hopfield_net.train(patterns)

# Función para cargar una imagen de entrada desde un archivo de texto
def load_input_image(file_path):
    with open(file_path, 'r') as f:
        image_str = f.read()
        image = np.array([int(bit) for bit in image_str])
    return image

# Ruta del archivo de la imagen de entrada que deseas reconocer
input_image_file = "circulo.txt"  # Reemplaza con la ruta de tu archivo

# Cargar la imagen de entrada desde el archivo de texto
input_image = load_input_image(input_image_file)

# Reconocimiento utilizando la red Hopfield
recognized_image = hopfield_net.update(input_image)

# Después de obtener la recognized_image desde la red, calcula la similitud con los patrones de entrenamiento
similarities = [np.dot(recognized_image, pattern) for pattern in patterns]

# Define un umbral para considerar que la imagen es similar a un patrón de entrenamiento
umbral_similitud = 1  # Ajusta este valor según tus necesidades

# Compara las similitudes con el umbral
similar = any(similarity >= umbral_similitud for similarity in similarities)

if similar:
    print("La imagen de entrada es similar a uno de los objetos entrenados.")
else:
    print("La imagen de entrada es diferente a los objetos entrenados.")

# Cargar y mostrar la imagen de entrada
input_image = load_input_image(input_image_file)
plt.figure(figsize=(5, 5))
plt.imshow(input_image.reshape(int(np.sqrt(len(input_image))), -1), cmap='gray')
plt.title("Imagen de Entrada")
plt.axis('off')
plt.show()

# Cargar y mostrar los patrones de entrenamiento
for i, pattern_file in enumerate(pattern_files):
    pattern = load_input_image(pattern_file)
    plt.figure(figsize=(5, 5))
    plt.imshow(pattern.reshape(int(np.sqrt(len(pattern))), -1), cmap='gray')
    plt.title(f"Patrón de Entrenamiento {i + 1}")
    plt.axis('off')
    plt.show()



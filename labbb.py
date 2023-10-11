import numpy as np

class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))

    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)  # Convierte el patrón en un vector columna
            self.weights += np.dot(pattern, pattern.T)
            np.fill_diagonal(self.weights, 0)  # La diagonal debe ser cero

    def energy(self, state):
        return -0.5 * np.dot(np.dot(state.T, self.weights), state)

    def update(self, state, max_iterations=100000000):
        for _ in range(max_iterations):
            new_state = np.sign(np.dot(self.weights, state))
            if np.array_equal(new_state, state):
                break
            state = new_state
        return state

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
pattern_files = ["pastoraleman1.txt", "pastoraleman2.txt","pastoraleman3.txt", "pastoraleman4.txt"]  # Agrega los nombres de tus archivos

# Tamaño del patrón (100x100)
pattern_size = 100 * 100

# Cargar patrones desde archivos de texto
patterns = load_patterns_from_files(pattern_files)

# Crear la red Hopfield y entrenarla
hopfield_net = HopfieldNetwork(pattern_size)
hopfield_net.train(patterns)

# Función para cargar una imagen de entrada desde un archivo de texto
def load_input_image(file_path):
    with open(file_path, 'r') as f:
        image_str = f.read()
        image = np.array([int(bit) for bit in image_str])
    return image

# Ruta del archivo de la imagen de entrada que deseas reconocer
input_image_file = "pastoraleman3.txt"  # Reemplaza con la ruta de tu archivo

# Cargar la imagen de entrada desde el archivo de texto
input_image = load_input_image(input_image_file)

# Impresión del patrón de entrada
print("Patrón de entrada:")
print(input_image.reshape(100, 100))

# Impresión de los patrones de entrenamiento
print("Patrones de entrenamiento:")
for i, pattern in enumerate(patterns):
    print(f"Patrón {i + 1}:")
    print(pattern.reshape(100, 100))

# Reconocimiento utilizando la red Hopfield
recognized_image = hopfield_net.update(input_image)

# Comparar el resultado con los patrones entrenados
similar = any(np.array_equal(recognized_image, pattern) for pattern in patterns)

if similar:
    print("La imagen de entrada es similar a uno de los objetos entrenados.")
else:
    print("La imagen de entrada es diferente a los objetos entrenados.")
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

    def update(self, state, max_iterations=100):
        for _ in range(max_iterations):
            new_state = np.sign(np.dot(self.weights, state))
            if np.array_equal(new_state, state):
                break
            state = new_state
        return state

# Ejemplo de uso
if __name__ == "__main__":
    pattern_size = 16  # Tamaño del patrón (por ejemplo, 4x4)
    patterns = [np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0])]  # Patrones de entrenamiento

    # Crear la red Hopfield y entrenarla
    hopfield_net = HopfieldNetwork(pattern_size)
    hopfield_net.train(patterns)

    # Imagen de prueba (puedes cambiarla)
    test_image = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Reconocimiento
    recognized_image = hopfield_net.update(test_image)
    print("Imagen de prueba:")
    print(test_image.reshape(4, 4))
    print("Imagen reconocida:")
    print(recognized_image.reshape(4, 4))





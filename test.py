import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.activation = lambda x: np.tanh(x)
        self.activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        
        # Inicializar los pesos y sesgos para cada capa
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            weight = np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i+1]))
            bias = np.random.uniform(-1, 1, (layer_sizes[i+1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, X):
        self.layer_inputs = [X]  # Lista para almacenar las entradas a cada capa
        self.layer_activations = [X]  # Lista para almacenar las activaciones de cada capa
        for weight, bias in zip(self.weights, self.biases):
            layer_input = np.dot(self.layer_activations[-1], weight) + bias
            activation = self.activation(layer_input)
            self.layer_inputs.append(layer_input)
            self.layer_activations.append(activation)
        return self.layer_activations[-1]

    def train(self, X, y, learning_rate=0.1, epochs=10000):
        for _ in range(epochs):
            for x, target in zip(X, y):
                output = self.forward(x)
                errors = [target - output]
                # Calcular el gradiente para la capa de salida
                d_output = errors[0] * self.activation_derivative(self.layer_activations[-1])
                deltas = [d_output]
                
                # Retropropagación del error
                for i in range(len(self.layer_activations) - 2, 0, -1):
                    delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(self.layer_activations[i])
                    deltas.insert(0, delta)
                
                # Actualizar los pesos y sesgos
                for i in range(len(self.weights)):
                    self.weights[i] += learning_rate * np.atleast_2d(self.layer_activations[i]).T * deltas[i]
                    self.biases[i] += learning_rate * deltas[i]

def predict(nn, X):
    predictions = np.array([nn.forward(x) for x in X])
    return np.round(predictions).flatten()


if __name__ == "__main__":
    # Datos para la compuerta XOR
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])


    layer_sizes = [2, 4, 1]  # 2 neuronas en la capa de entrada, dos capas ocultas de 4 neuronas, 1 neurona en la capa de salida
    nn = NeuralNetwork(layer_sizes)
    nn.train(X_xor, y_xor, learning_rate=0.1, epochs=10000)


    # Generar una malla de puntos para visualizar el límite de decisión
    x1_range = np.linspace(-0.5, 1.5, 1000)
    x2_range = np.linspace(-0.5, 1.5, 1000)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    grid = np.c_[xx1.ravel(), xx2.ravel()]

    # Predicciones para cada punto en la malla
    predictions_grid = predict(nn, grid).reshape(xx1.shape)

    # Graficar el límite de decisión y los puntos de entrenamiento
    plt.figure(figsize=(8, 8))
    plt.contourf(xx1, xx2, predictions_grid, alpha=0.7, levels=[-1, 0, 1])
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=100, edgecolor='k', marker='o')
    plt.title('Límite de Decisión de la Red Neuronal para XOR')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.grid(True)
    plt.show()


    # Generar una malla de puntos para visualizar el límite de decisión
    x1_range = np.linspace(-10, 10, 1000)
    x2_range = np.linspace(-10, 10, 1000)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    grid = np.c_[xx1.ravel(), xx2.ravel()]

    # Predicciones para cada punto en la malla
    predictions_grid = predict(nn, grid).reshape(xx1.shape)

    # Graficar el límite de decisión y los puntos de entrenamiento
    plt.figure(figsize=(8, 8))
    plt.contourf(xx1, xx2, predictions_grid, alpha=0.7, levels=[-1, 0, 1])
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=100, edgecolor='k', marker='o')
    plt.title('Límite de Decisión de la Red Neuronal para XOR')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.grid(True)
    plt.show()
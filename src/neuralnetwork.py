import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, activation="ReLU"):
        self.layers = [input_size] + [hidden_size] * hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(self.layers)-1)]
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]
        self.activation = activation

    def activation_func(self, x):
        if self.activation == "ReLU":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)

    def forward(self, X):
        self.a = [X]
        for W, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], W) + b
            self.a.append(self.activation_func(z))
        return self.a[-1]
import numpy as np
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, activation="ReLU", init_method = "random", loss_type = "mse"):
        self.layers = [input_size] + [hidden_size] * hidden_layers + [output_size]
        self.init_method = init_method
        self.weights = self.initialize_weights()
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers)-1)]
        self.activation = activation
        self.loss_type = loss_type
    
    def initialize_weights(self):
        weights = []
        for i in range(len(self.layers) - 1):
            if self.init_method == "Xavier":
                limit = np.sqrt(1 / self.layers[i])
                weights.append(np.random.uniform(-limit, limit, (self.layers[i], self.layers[i+1])))
            else:
                weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.01)
        return weights
    
    def activation_func(self, x, is_output = False):
        if is_output and self.loss_type == "cross_entropy":
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        if self.activation == "ReLU":
            return np.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return np.tanh(x)

    def forward(self, X):
        self.a = [X]
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(self.a[-1], W) + b
            # Check if this is the output layer
            is_output = (i == len(self.weights) - 1)
            self.a.append(self.activation_func(z, is_output=is_output))
        return self.a[-1]
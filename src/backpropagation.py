import numpy as np

class Backpropagation:
    def __init__(self, model, optimizer="sgd", learning_rate=0.01, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        """
        Initialize the backpropagation optimizer.
        :param model: NeuralNetwork model instance
        :param optimizer: Optimization algorithm ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam')
        :param learning_rate: Learning rate for weight updates
        :param momentum: Momentum factor for momentum-based optimizers
        :param beta: Decay factor for RMSprop
        :param beta1: First moment decay rate (Adam, Nadam)
        :param beta2: Second moment decay rate (Adam, Nadam)
        :param epsilon: Small value to prevent division by zero
        :param weight_decay: L2 Regularisation
        """
        self.model = model
        self.optimizer = optimizer
        self.lr = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # Initialize velocity (for Momentum and Nesterov)
        self.velocities = [np.zeros_like(W) for W in model.weights]

        # Initialize moving averages (for Adam, Nadam, RMSprop)
        self.m = [np.zeros_like(W) for W in model.weights]
        self.v = [np.zeros_like(W) for W in model.weights]
        self.t = 0  # Time step for Adam/Nadam
    
    def backward(self, y_true, y_pred):
        """
        Perform backward pass to compute gradients.
        :param y_true: True labels
        :param y_pred: Predicted output
        :return: Gradients for each layer
        """
        gradients = []
        delta = y_pred - y_true
        
        for i in reversed(range(len(self.model.weights))):
            grad_W = np.dot(self.model.a[i].T, delta) / y_true.shape[0]
            grad_W += self.weight_decay * self.model.weights[i]  # Apply L2 regularization
            gradients.insert(0, grad_W)
            
            if i > 0:
                if self.model.activation == "ReLU":
                    delta = np.dot(delta, self.model.weights[i].T) * (self.model.a[i] > 0)
                elif self.model.activation == "sigmoid":
                    delta = np.dot(delta, self.model.weights[i].T) * (self.model.a[i] * (1 - self.model.a[i]))
                elif self.model.activation == "tanh":
                    delta = np.dot(delta, self.model.weights[i].T) * (1 - self.model.a[i]**2)
    
        
        return gradients
    
    def update_weights(self, gradients):
        """
        Updates the model weights using the selected optimizer.
        :param gradients: List of gradient matrices for each layer
        """
        self.t += 1  # Increment time step for Adam/Nadam

        for i in range(len(self.model.weights)):
            grad = gradients[i] + self.weight_decay * self.model.weights[i]  # Apply L2 regularization

            if self.optimizer == "sgd":
                # Stochastic Gradient Descent (SGD)
                self.model.weights[i] -= self.lr * grad

            elif self.optimizer == "momentum":
                # Momentum-Based Gradient Descent
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                self.model.weights[i] += self.velocities[i]

            elif self.optimizer == "nesterov":
                # Nesterov Accelerated Gradient Descent
                prev_velocity = self.velocities[i]
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                self.model.weights[i] += -self.momentum * prev_velocity + (1 + self.momentum) * self.velocities[i]

            elif self.optimizer == "rmsprop":
                # RMSprop
                self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (grad ** 2)
                self.model.weights[i] -= self.lr * grad / (np.sqrt(self.v[i]) + self.epsilon)

            elif self.optimizer == "adam":
                # Adam (Adaptive Moment Estimation)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                self.model.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            elif self.optimizer == "nadam":
                # Nadam (Nesterov-accelerated Adam)
                m_hat = (self.beta1 * self.m[i] + (1 - self.beta1) * grad) / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                self.model.weights[i] -= self.lr * (self.beta1 * m_hat + (1 - self.beta1) * grad / (1 - self.beta1 ** self.t)) / (np.sqrt(v_hat) + self.epsilon)

            else:
                raise ValueError(f"Optimizer '{self.optimizer}' is not recognized.")

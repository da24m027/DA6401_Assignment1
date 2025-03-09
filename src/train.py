import argparse
import wandb
from neuralnetwork import NeuralNetwork
from backpropagation import Backpropagation
from utils import *
from keras.datasets import mnist,fashion_mnist

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network with given parameters.")

    # Define all arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name for Weights & Biases.")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb entity.")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset to use.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", help="Loss function.")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam", help="Optimizer.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum for certain optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for rmsprop.")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for adam and nadam.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for adam and nadam.")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8, help="Epsilon for optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier", help="Weight initialization method.")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=64, help="Number of neurons per hidden layer.")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", help="Activation function.")

    return parser.parse_args()

def train(config):
    wandb.init(entity=config.wandb_entity, project=config.wandb_project)

    if config.dataset=="mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)

    val_size = int(0.1 * len(X_train))
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train, y_train = X_train[val_size:], y_train[val_size:]

    model = NeuralNetwork(input_size=784, hidden_layers=config.num_layers, hidden_size=config.hidden_size, output_size=10, activation=config.activation, init_method=config.weight_init, loss_type = config.loss)
    backprop = Backpropagation(model, optimizer=config.optimizer, learning_rate=config.learning_rate, momentum=config.momentum, beta=config.beta, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon, weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
      loss = 0
      for i in range(0, len(X_train), config.batch_size):
        batch_X = X_train[i:i + config.batch_size]
        batch_y = y_train[i:i + config.batch_size]

        # Forward & Backward pass
        y_pred = model.forward(batch_X)
        gradients = backprop.backward(batch_y, y_pred)
        backprop.update_weights(gradients)

        # Compute loss
        loss += calculate_loss(batch_y, y_pred, config.loss)

      loss /= len(X_train)
      wandb.log({"epoch": epoch+1, "loss": loss})

      # Validate model
      val_pred = model.forward(X_val)
      val_loss = calculate_loss(y_val, val_pred, config.loss)
      val_accuracy = calculate_accuracy(y_val, val_pred)
      wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_accuracy": val_accuracy})

      print(f"Epoch {epoch+1}/{config.epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    test_pred = model.forward(X_test)
    test_accuracy = calculate_accuracy(y_test, test_pred)
    wandb.log({"test_accuracy": test_accuracy})
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    args = get_args()
    train(args)
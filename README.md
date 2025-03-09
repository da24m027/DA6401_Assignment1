# DA6401_Assignment1
Assignment 1 Submission for DA6401.

Y. Vishwambhar Reddy (DA24M027)

WandB Report can be accessed [here](https://wandb.ai/da24m027-indian-institute-of-technology-madras/DA6401_Assignment1/reports/DA6401-Assignment-1--VmlldzoxMTcwNTc1NA?accessToken=zldlkrwtbxkfx86gxbq7kssucu7zi2fgn7trx2igthehjg770apcajjh8fmad8rh)

## Project Structure
```
DA6401_Assignment1/
├── src/                 # Source code
│   ├── neuralnetwork.py         # Neural network implementation
│   ├── backpropagation.py      # Backpropagation logic
│   ├── train.py         # Training script(main entry point)
│   ├── utils.py         # Helper functions
│   ├── Data_Visualization_Wandb.py     # Code for Question 1 (visualizing dataset)
│   ├── Wandb_Experiments.ipynb     # Code for Question 4,5,6,8 
│   ├── Best_Model.ipynb     # Code for Question 7 
│   ├── MNIST.ipynb     # Code for Question 10    
├── results/             # Store experiment results
├── requirements.txt     # Dependencies
├── README.md            # Setup and usage instructions
```

## Training Scripy
The `train.py` script can be used to train the model and allows users to specify various training parameters via command-line arguments.

### Usage
Login to Wandb from the terminal
```sh
wandb.login()
```
Then run the training script with the desired hyperparameter configuration and the wandb project name where the run will be logged
```sh
python train.py [OPTIONS]
```

## Question 1
The visualization code of dataset images through WandB can be accessed [here](https://github.com/da24m027/DA6401_Assignment1/blob/main/src/Data_Visualization_Wandb.ipynb)

## Question 2
The feedforward neural network is implemented by the `NeuralNetwork` class in `neuralnetwork.py`. The implementation allows for customization of network architecture, activation functions, weight initialization methods, and loss functions.

### Features
- Configurable network architecture with variable hidden layers and neurons
- Multiple activation function options (ReLU, sigmoid, tanh)
- Weight initialization methods (random, Xavier)
- Support for different loss functions (MSE, cross-entropy)

### Usage
```python
import numpy as np
from neuralnetwork import NeuralNetwork

# Create a neural network with:
# - 4 input features
# - 2 hidden layers
# - 8 neurons per hidden layer
# - 3 output classes
network = NeuralNetwork(
    input_size=4,
    hidden_layers=2,
    hidden_size=8,
    output_size=3,
    activation="ReLU",
    init_method="Xavier",
    loss_type="cross_entropy"
)

# Forward pass
X = np.random.randn(10, 4)  # 10 samples with 4 features each
predictions = network.forward(X)
```
## Question 3
The backpropagation framework is implemented by the `Backpropagation` class in `backpropagation.py`. The implementation allows for customization of optimizer.

### Features
- Multiple optimization algorithms (`sgd`, `momentum`, `nesterov`, `rmsprop`, `adam`, `nadam`).
- Weight regularization via L2 decay.

### Usage
```python
from backpropagation import Backpropagation

optimizer = Backpropagation(model=nn, optimizer="adam", learning_rate=0.01)
gradients = optimizer.backward(y_true, y_pred)  # Compute gradients
optimizer.update_weights(gradients)  # Update weights
```
### Question 4,5,6,7,8
The Questions 4,5,6 and 8 were implemented in `Wandb_Experiments.ipynb` and Question 7 was implemented in `BestModel.ipynb`

### Question 10
Since the model was trained on Fashion MNIST, the top performing hyperparameter configurations can be used for MNIST also. This is MNIST is a similar, and a much simpler image classification task with the same number of output classes. The top 3 configurations on Fashion-MNIST were chosen for training on MNIST dataset, the configurations are

```
config1 = {
    "epochs":10,
    "activation":"sigmoid",
    "batch_size":16,
    "hidden_layers":4,
    "hidden_size":64,
    "learning_rate":0.001,
    "optimizer":"adam",
    "weights":"Xavier",
    "weight_decay":0,
    "loss_type":"cross_entropy"
}

config2 = {
    "epochs":5,
    "activation":"tanh",
    "batch_size":32,
    "hidden_layers":3,
    "hidden_size":64,
    "learning_rate":0.001,
    "optimizer":"adam",
    "weights":"Xavier",
    "weight_decay":0.0005,
    "loss_type":"cross_entropy"
}

config3 = {
    "epochs":5,
    "activation":"ReLU",
    "batch_size":16,
    "hidden_layers":3,
    "hidden_size":128,
    "learning_rate":0.001,
    "optimizer":"rmsprop",
    "weights":"Xavier",
    "weight_decay":0,
    "loss_type":"mse"
}
```
The test accuracy on MNIST dataset is 96.79%. The code can be accessed [here](https://github.com/da24m027/DA6401_Assignment1/blob/main/src/MNIST.ipynb)

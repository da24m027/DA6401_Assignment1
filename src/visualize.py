import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load data
(X_train, y_train), (_, _) = fashion_mnist.load_data()

# Define class names
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Plot one image per class
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[y_train == i][0], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')

plt.savefig("../results/fashion_mnist_samples.png")  # Save figure
plt.show()
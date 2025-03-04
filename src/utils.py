import numpy as np

def preprocess_data(X_train, y_train, X_test, y_test):
    """
    Normalize and reshape dataset.
    """
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize
    X_train, X_test = X_train.reshape(-1, 784), X_test.reshape(-1, 784)  # Flatten images

    # One-hot encoding
    y_train_encoded = np.zeros((y_train.size, 10))
    y_train_encoded[np.arange(y_train.size), y_train] = 1

    y_test_encoded = np.zeros((y_test.size, 10))
    y_test_encoded[np.arange(y_test.size), y_test] = 1

    return X_train, y_train_encoded, X_test, y_test_encoded

def evaluate(model, X_test, y_test):
    """
    Evaluate model accuracy.
    """
    y_pred = model.forward(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_labels == y_true_labels)
    return accuracy
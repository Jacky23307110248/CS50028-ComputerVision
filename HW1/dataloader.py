import numpy as np
import gzip
import os

def load_mnist(data_path="./FashionMNIST/raw"):
    path = data_path
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Data path not found: {path}. "
            "Please place Fashion-MNIST raw files under this path or pass correct --data_path."
        )
    # read train images
    with gzip.open(os.path.join(path, "train-images-idx3-ubyte.gz"), "rb") as f:
        X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    # read train labels
    with gzip.open(os.path.join(path, "train-labels-idx1-ubyte.gz"), "rb") as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    # read test images
    with gzip.open(os.path.join(path, "t10k-images-idx3-ubyte.gz"), "rb") as f:
        X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
    # read test labels
    with gzip.open(os.path.join(path, "t10k-labels-idx1-ubyte.gz"), "rb") as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    # Normalization
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # one-hot encoding
    y_train_onehot = np.eye(10, dtype=np.float32)[y_train]
    y_test_onehot = np.eye(10, dtype=np.float32)[y_test]

    return X_train, y_train_onehot, X_test, y_test_onehot, y_train, y_test
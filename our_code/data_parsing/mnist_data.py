from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
import networkx as nx
import numpy as np


def get_digit_data():
    """
    Load and preprocess MNIST Digit data.

    Return:
        (x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_names): numpy arrays and one list
    """
    # Load the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Shuffle the training data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    # Shuffle the test data.
    permutation = np.random.permutation(x_test.shape[0])
    x_test = x_test[permutation]
    y_test = y_test[permutation]
    # Create the one-hot versions of y
    y_train_one_hot = np.zeros((x_train.shape[0], 10))
    y_test_one_hot = np.zeros((x_test.shape[0], 10))
    for i, y in enumerate(y_train):
        y_train_one_hot[i][y] = 1
    for i, y in enumerate(y_test):
        y_test_one_hot[i][y] = 1
    class_names = [str(i) for i in range(10)]
    return (
        x_train,
        y_train,
        y_train_one_hot,
        x_test,
        y_test,
        y_test_one_hot,
        class_names,
    )


def get_parity_tree():
    """
    Create a hierarchical tree from origin to even/odd to respective digits.
    This reflects the true hierarchy of the data.

    Return:
        tree: nx.Digraph
    """
    tree = nx.DiGraph()
    for label in [0, 2, 4, 6, 8]:
        tree.add_edge("even", str(label))
        tree.add_edge("odd", str(label + 1))
    tree.add_edge("origin", "even")
    tree.add_edge("origin", "odd")
    nodes = list(tree.nodes)
    for node in nodes:
        tree.add_node(node, name=node)
    return tree


# Label 	Description
# 0 	    T-shirt/top
# 1 	    Trouser
# 2 	    Pullover
# 3 	    Dress
# 4 	    Coat
# 5 	    Sandal
# 6 	    Shirt
# 7 	    Sneaker
# 8 	    Bag
# 9 	    Ankle boot
def get_fashion_data():
    """
    Load and preprocess MNIST Digit data.

    Return:
        (x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_names): numpy arrays and one list
    """

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Shuffle the training data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    # Shuffle the test data.
    permutation = np.random.permutation(x_test.shape[0])
    x_test = x_test[permutation]
    y_test = y_test[permutation]
    # Create the one-hot versions of y
    y_train_one_hot = np.zeros((x_train.shape[0], 10))
    y_test_one_hot = np.zeros((x_test.shape[0], 10))
    for i, y in enumerate(y_train):
        y_train_one_hot[i][y] = 1
    for i, y in enumerate(y_test):
        y_test_one_hot[i][y] = 1
    return (
        x_train,
        y_train,
        y_train_one_hot,
        x_test,
        y_test,
        y_test_one_hot,
        class_names,
    )


def get_fashion_tree():
    """
    Create a hierarchical tree from origin to even/odd to respective digits.
    This reflects the true hierarchy of the data.

    Return:
        tree: nx.Digraph
    """

    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    high_level = ["shoes", "top", "fancy"]
    class_to_group_mapping = {
        0: 1,
        1: 2,
        2: 1,
        3: 2,
        4: 1,
        5: 0,
        6: 1,
        7: 0,
        8: 2,
        9: 0,
    }
    tree = nx.DiGraph()
    for low, high in class_to_group_mapping.items():
        tree.add_edge(high_level[high], class_names[low])
    for high in high_level:
        tree.add_edge("origin", high)
    nodes = list(tree.nodes)
    for node in nodes:
        tree.add_node(node, name=node)
    return tree


def make_noisy(x, noise_level=0.5):
    """
    Add normal noise to the data.

    Params:
        x: numpy array
        noise_level: float

    Return:
        tree: nx.Digraph
    """
    noise = np.random.normal(loc=0, scale=noise_level, size=x.shape).astype("float32")
    x_noisy = np.clip(x + noise, 0, 1)
    return x_noisy

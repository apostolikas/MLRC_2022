import networkx as nx
import numpy as np
import torchvision.datasets as datasets


def to_categorical(y, num_classes=None, dtype="float32"):
    """
    Convert labels to categorical.

    Params:
        y: list of integers
        num_classes: None or integer
        dtype: string

    Return:
        categorical: numpy array
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


cifar_fine_labels = [
    "apple",  # id 0
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "computer_keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

cifar_2_labels = [
    "sea-creatures",
    "mammals",
    "large natural outdoor scenes",
    "large man-made outdoor scenes",
    "vehicles",
    "items",
    "plants",
    "non-mammals",
]

cifar_3_labels = [
    "animals",
    "large natural outdoor scenes",
    "artificial objects",
    "plants",
]

cifar_4_labels = ["living things", "non-living things"]


def sparse2coarse(targets):
    """
    Convert sparse to coarse labels for specific targets.

    Params:
        targets: list of integers

    Return:
        coarse labels: numpy array
    """
    coarse_labels = np.array(
        [
            4,
            1,
            14,
            8,
            0,
            6,
            7,
            7,
            18,
            3,
            3,
            14,
            9,
            18,
            7,
            11,
            3,
            9,
            7,
            11,
            6,
            11,
            5,
            10,
            7,
            6,
            13,
            15,
            3,
            15,
            0,
            11,
            1,
            10,
            12,
            14,
            16,
            9,
            11,
            5,
            5,
            19,
            8,
            8,
            15,
            13,
            14,
            17,
            18,
            10,
            16,
            4,
            17,
            4,
            2,
            0,
            17,
            4,
            18,
            17,
            10,
            3,
            2,
            12,
            12,
            16,
            12,
            1,
            9,
            19,
            2,
            10,
            0,
            1,
            16,
            12,
            9,
            13,
            15,
            13,
            16,
            19,
            2,
            4,
            6,
            19,
            5,
            5,
            8,
            19,
            18,
            1,
            2,
            15,
            6,
            0,
            17,
            8,
            14,
            13,
        ]
    )
    return coarse_labels[targets]


def get_deep_data():
    """
    Load CIFAR 100 dataset splits with 5 layers of hierarchy.
    """
    # (x_train, y_train0), (x_test, y_test0) = cifar100.load_data(label_mode='fine')
    # (_, y_train1), (_, y_test1) = cifar100.load_data(label_mode='coarse')

    cifar_trainset = datasets.CIFAR100(
        root="./data", train=True, download=True
    )  # ,transform=data_transforms)
    cifar_testset = datasets.CIFAR100(
        root="./data", train=False, download=True
    )  # ,transform=data_transforms)

    x_train = np.zeros(
        [50000, 32, 32, 3]
    )  # data_shape[0],data_shape[1],data_shape[2]])
    y_train0 = np.zeros([50000, 1])
    for i in range(len(cifar_trainset)):
        x_train[i, :, :] = cifar_trainset[i][0]
        y_train0[i] = cifar_trainset[i][1]

    x_test = np.zeros([10000, 32, 32, 3])
    y_test0 = np.zeros([10000, 1])
    for i in range(len(cifar_testset)):
        x_test[i, :, :] = cifar_testset[i][0]
        y_test0[i] = cifar_testset[i][1]

    cifar_trainset.targets = sparse2coarse(cifar_trainset.targets)
    cifar_testset.targets = sparse2coarse(cifar_testset.targets)

    y_train1 = np.zeros([50000, 1])
    for i in range(len(cifar_trainset)):
        y_train1[i] = cifar_trainset[i][1]

    y_test1 = np.zeros([10000, 1])
    for i in range(len(cifar_testset)):
        y_test1[i] = cifar_testset[i][1]

    image_size = x_train.shape[1]
    original_dim = image_size * image_size

    # Shuffle the training data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train0 = y_train0[permutation]
    y_train1 = y_train1[permutation]
    # Shuffle the test data.
    permutation = np.random.permutation(x_test.shape[0])
    x_test = x_test[permutation]
    y_test0 = y_test0[permutation]
    y_test1 = y_test1[permutation]
    y_train0_one_hot = to_categorical(y_train0)
    y_train1_one_hot = to_categorical(y_train1)
    y_test0_one_hot = to_categorical(y_test0)
    y_test1_one_hot = to_categorical(y_test1)

    # Label the higher levels of the tree.
    (
        (level1_mapping, _, _, _),
        (_, fine_to_level2, fine_to_level3, fine_to_level4),
        _,
    ) = get_deep_tree()
    coarse_names = list(level1_mapping.keys())
    # For train data.
    y_train2 = np.zeros_like(y_train0)
    y_train3 = np.zeros_like(y_train0)
    y_train4 = np.zeros_like(y_train0)
    for y_idx, fine_y in enumerate(y_train0):
        fine_y_val = int(fine_y[0])
        fine_y_label = cifar_fine_labels[fine_y_val]
        level2_y = fine_to_level2.get(fine_y_label)
        level3_y = fine_to_level3.get(fine_y_label)
        level4_y = fine_to_level4.get(fine_y_label)
        assert level2_y is not None and level3_y is not None and level4_y is not None
        level2_val = cifar_2_labels.index(level2_y)
        level3_val = cifar_3_labels.index(level3_y)
        level4_val = cifar_4_labels.index(level4_y)
        y_train2[y_idx] = level2_val
        y_train3[y_idx] = level3_val
        y_train4[y_idx] = level4_val
    # And convert to categorical
    y_train2_one_hot = to_categorical(y_train2)
    y_train3_one_hot = to_categorical(y_train3)
    y_train4_one_hot = to_categorical(y_train4)
    # For test data.
    y_test2 = np.zeros_like(y_test0)
    y_test3 = np.zeros_like(y_test0)
    y_test4 = np.zeros_like(y_test0)
    for y_idx, fine_y in enumerate(y_test0):
        fine_y_val = int(fine_y[0])
        fine_y_label = cifar_fine_labels[fine_y_val]
        level2_y = fine_to_level2.get(fine_y_label)
        level3_y = fine_to_level3.get(fine_y_label)
        level4_y = fine_to_level4.get(fine_y_label)
        assert level2_y is not None and level3_y is not None and level4_y is not None
        level2_val = cifar_2_labels.index(level2_y)
        level3_val = cifar_3_labels.index(level3_y)
        level4_val = cifar_4_labels.index(level4_y)
        y_test2[y_idx] = level2_val
        y_test3[y_idx] = level3_val
        y_test4[y_idx] = level4_val
    # And convert to categorical
    y_test2_one_hot = to_categorical(y_train2)
    y_test3_one_hot = to_categorical(y_train3)
    y_test4_one_hot = to_categorical(y_train4)
    return (
        x_train,
        y_train0,
        y_train0_one_hot,
        y_train1,
        y_train1_one_hot,
        y_train2,
        y_train2_one_hot,
        y_train3,
        y_train3_one_hot,
        y_train4,
        y_train4_one_hot,
        x_test,
        y_test0,
        y_test0_one_hot,
        y_test1,
        y_test1_one_hot,
        y_test2,
        y_test2_one_hot,
        y_test3,
        y_test3_one_hot,
        y_test4,
        y_test4_one_hot,
        [
            cifar_fine_labels,
            coarse_names,
            cifar_2_labels,
            cifar_3_labels,
            cifar_4_labels,
        ],
    )


def get_deep_tree():  # same method
    """
    Create tree of hierarchy and mappings of the labels for each level.
    """
    level_1_mapping = {
        "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "food containers": ["bottle", "bowl", "can", "cup", "plate"],
        "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "household electrical device": [
            "clock",
            "computer_keyboard",
            "lamp",
            "telephone",
            "television",
        ],
        "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large man-made outdoor things": [
            "bridge",
            "castle",
            "house",
            "road",
            "skyscraper",
        ],
        "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
        "large omnivores and herbivores": [
            "camel",
            "cattle",
            "chimpanzee",
            "elephant",
            "kangaroo",
        ],
        "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    }
    level_2_mapping = {
        "sea-creatures": ["fish", "aquatic mammals"],
        "mammals": [
            "people",
            "small mammals",
            "medium-sized mammals",
            "large omnivores and herbivores",
            "large carnivores",
        ],
        "large natural outdoor scenes": ["large natural outdoor scenes"],
        "large man-made outdoor scenes": ["large man-made outdoor things"],
        "vehicles": ["vehicles 1", "vehicles 2"],
        "items": [
            "household electrical device",
            "household furniture",
            "food containers",
        ],
        "plants": ["fruit and vegetables", "flowers", "trees"],
        "non-mammals": ["insects", "reptiles", "non-insect invertebrates"],
    }
    level_3_mapping = {
        "animals": ["non-mammals", "sea-creatures", "mammals"],
        "large natural outdoor scenes": ["large natural outdoor scenes"],
        "artificial objects": ["large man-made outdoor scenes", "vehicles", "items"],
        "plants": ["plants"],
    }
    level_4_mapping = {
        "living things": ["plants", "animals"],
        "non-living things": ["large natural outdoor scenes", "artificial objects"],
    }

    recovered_fine = set()
    for key4, items3 in level_4_mapping.items():
        for item3 in items3:
            # print("Fetching items for key", item3)
            items2 = level_3_mapping.get(item3)
            for item2 in items2:
                # print("Fetching items for key", item2)
                items1 = level_2_mapping.get(item2)
                for item1 in items1:
                    # print("Fetching items for key", item1)
                    items0 = level_1_mapping.get(item1)
                    for item0 in items0:
                        recovered_fine.add(item0)
    assert sum([len(items) for items in level_1_mapping.values()]) == len(
        recovered_fine
    )

    # Create the reverse maps
    fine_to_1 = {}
    fine_to_2 = {}
    fine_to_3 = {}
    fine_to_4 = {}
    for level1, level0s in level_1_mapping.items():
        for level0 in level0s:
            fine_to_1[level0] = level1
    for fine_label in recovered_fine:
        level1 = fine_to_1.get(fine_label)
        for level2, level1s in level_2_mapping.items():
            if level1 in level1s:
                fine_to_2[fine_label] = level2
                break
    for fine_label in recovered_fine:
        level2 = fine_to_2.get(fine_label)
        for level3, level2s in level_3_mapping.items():
            if level2 in level2s:
                fine_to_3[fine_label] = level3
                break
    for fine_label in recovered_fine:
        level3 = fine_to_3.get(fine_label)
        for level4, level3s in level_4_mapping.items():
            if level3 in level3s:
                fine_to_4[fine_label] = level4
                break

    # Create the actual tree
    tree = nx.DiGraph()
    for level1, level0s in level_1_mapping.items():
        for level0 in level0s:
            tree.add_edge(level1, level0)
    for level2, level1s in level_2_mapping.items():
        for level1 in level1s:
            tree.add_edge(level2, level1)
    for level3, level2s in level_3_mapping.items():
        for level2 in level2s:
            tree.add_edge(level3, level2)
    for level4, level3s in level_4_mapping.items():
        for level3 in level3s:
            tree.add_edge(level4, level3)
    for level4 in level_4_mapping.keys():
        tree.add_edge("origin", level4)
    nodes = list(tree.nodes)
    for node in nodes:
        tree.add_node(node, name=node)

    return (
        (level_1_mapping, level_2_mapping, level_3_mapping, level_4_mapping),
        (fine_to_1, fine_to_2, fine_to_3, fine_to_4),
        tree,
    )

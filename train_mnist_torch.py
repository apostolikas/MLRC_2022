import numpy as np
import torch
from our_code.data_parsing.mnist_data import get_digit_data, make_noisy, get_parity_tree
from our_code.utils.metrics import trees_match, graph_edit_dist
from our_code.data_parsing.datasets import Dataset
import torch.utils.data as data
from our_code.utils.train import train
from our_code.utils.eval import eval_label_parity
from our_code.utils.metrics import get_mst
from our_code.models.proto_model import ProtoModel
import networkx as nx
from our_code.utils.seeds import set_seed

# import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(torch.cuda.is_available())

set_seed(0)

batch_size = 128  # he hardcoded below. I put it here for clarity
latent_dim = 32
noise_level = 0.0  # If you want to play with this, can blur images, but don't need to.
use_digit_and_parity = True
digit_only = False
parity_only = False

(
    x_train,
    y_train,
    y_train_one_hot,
    x_test,
    y_test,
    y_test_one_hot,
    class_labels,
) = get_digit_data()
# x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot, class_labels = get_fashion_data()

x_train_noisy = make_noisy(x_train, noise_level=noise_level)
x_test_noisy = make_noisy(x_test, noise_level=noise_level)


parity_train_one_hot = np.zeros((y_train.shape[0], 2))

# I set the dtype to int64 maybe we can set it to unint8
parity_train = np.zeros((y_train.shape[0]), dtype=np.int64)
for i, y in enumerate(y_train):
    parity_train[i] = (y % 2)
    parity_train_one_hot[i][y % 2] = 1

# I set the dtype to int64 maybe we can set it to unint8
parity_test = np.zeros((y_test.shape[0]), dtype=np.int64)
parity_test_one_hot = np.zeros((y_test.shape[0], 2))
for i, y in enumerate(y_test):
    parity_test[i] = y % 2
    parity_test_one_hot[i][y % 2] = 1

if use_digit_and_parity:
    output_sizes = [10, 2]
    output = [y_test, parity_test]
    train_dataset = Dataset(x_train, x_train_noisy, y_train, parity_train)

    test_dataset = Dataset(x_test, x_test_noisy, y_test, parity_test)

train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False)
ground_truth_tree = get_parity_tree()


classification_weights = (
    [10] if not use_digit_and_parity else [10, 10]
)  # Mess with these weights as desired.
proto_dist_weights = (
    [1] if not use_digit_and_parity else [1, 1]
)  # How realistic are the prototypes
feature_dist_weights = (
    [1] if not use_digit_and_parity else [0.1, 100]
)  # How close to prototypes are embeddings (cluster size)
disentangle_weights = [[0 for _ in range(2)] for _ in range(2)]
disentangle_weights[0] = [0, -10]
kl_losses = [1] if not use_digit_and_parity else [0, 0]
duplication_factors = [1] if not use_digit_and_parity else [1, 1]

all_proto_grids = [None, None]

all_ac = []
all_ed = []
all_seeds_hier_accs = [[] for i in range(len(output_sizes))]
# Run a bunch of trials.


for model_id in range(10):
    set_seed(model_id)


    proto_model = ProtoModel(
        output_sizes,
        duplication_factors=duplication_factors,
        input_size=784,
        classification_weights=classification_weights,
        proto_dist_weights=proto_dist_weights,
        feature_dist_weights=feature_dist_weights,
        disentangle_weights=disentangle_weights,
        kl_losses=kl_losses,
        latent_dim=latent_dim,
        proto_grids=all_proto_grids,
        in_plane_clusters=True,
        use_shadow_basis=False,
        align_fn=torch.mean,
        network_type="dense_mnist",
    )
    proto_model = proto_model.to(device)

    train(proto_model, 20, train_dataloader)
    
    tmp_acc_lists, tmp_average_cost = eval_label_parity(
        proto_model, test_dataloader, gold_tree=(ground_truth_tree, class_labels)
    )

    for i, cur_seed_hier_accs in enumerate(all_seeds_hier_accs):
        cur_seed_hier_accs.append(tmp_acc_lists[i])

    # proto_model.train(x_train_noisy, x_train, one_hot_output, batch_size=128, epochs=1)
    # proto_model.save('../saved_models/mnist_models/')
    mst = get_mst(
        proto_model, add_origin=True, plot=False, labels=[class_labels, ["even", "odd"]]
    )
    tree_matches = trees_match(mst, ground_truth_tree)
    print("Tree matches", tree_matches)
    # Super fast if the trees are actually close together.
    edit_dist = graph_edit_dist(ground_truth_tree, mst)
    print("Edit distance", edit_dist)
    all_ac.append(tmp_average_cost)
    all_ed.append(edit_dist)
    # If you want to visualize the latent space, uncomment below. I'd recommend training a model with latent dim 2 for
    # the best visualizations.


for i, cur_seed_hier_accs in enumerate(all_seeds_hier_accs):
    print(
        "For all the seeds: The accuracy for level ",
        str(i),
        "is",
        np.mean(cur_seed_hier_accs),
        "±",
        np.std(cur_seed_hier_accs),
    )

print("For all the seeds: The average cost is ", np.mean(all_ac), "±", np.std(all_ac))
print("For all the seeds: The edit distance is ", np.mean(all_ed), "±", np.std(all_ed))

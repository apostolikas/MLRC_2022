import numpy as np
import torch
import torch.utils.data as data
from our_code.data_parsing.mnist_data import (
    get_fashion_data,
    make_noisy,
    get_fashion_tree,
)
from our_code.utils.metrics import trees_match, graph_edit_dist
from our_code.utils.train import train
from our_code.utils.eval import eval_label_parity  # , eval_proto_diffs_fashion
from our_code.utils.metrics import get_mst
from our_code.models.proto_model import ProtoModel
from our_code.data_parsing.datasets import Dataset
from our_code.utils.seeds import set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(torch.cuda.is_available())

batch_size = 64
latent_dim = 32
noise_level = 0.0
use_class_and_group = True
class_only = False
group_only = False
(
    x_train,
    y_train,
    y_train_one_hot,
    x_test,
    y_test,
    y_test_one_hot,
    class_labels,
) = get_fashion_data()
ground_truth_tree = get_fashion_tree()

x_train_noisy = make_noisy(x_train, noise_level=noise_level)
x_test_noisy = make_noisy(x_test, noise_level=noise_level)
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
# Groups:
# 0         Shoes (5, 7, 9)
# 1         Top (0, 2, 4, 6)
# 2         Fancy (1, 3, 8)
class_to_group_mapping = {0: 1, 1: 2, 2: 1, 3: 2, 4: 1, 5: 0, 6: 1, 7: 0, 8: 2, 9: 0}

num_groups = len(set(class_to_group_mapping.values()))
group_train_one_hot = np.zeros((y_train.shape[0], num_groups))
# I set the dtype to int64 maybe we can set it to unint8
group_train = np.zeros((y_train.shape[0]), dtype=np.int64)
for i, y in enumerate(y_train):
    group_train_one_hot[i][class_to_group_mapping.get(y)] = 1
    group_train[i] = class_to_group_mapping.get(y)

group_test = np.zeros((y_test.shape[0]))
group_test_one_hot = np.zeros((y_test.shape[0], num_groups))
for i, y in enumerate(y_test):
    group_test[i] = class_to_group_mapping.get(y)
    group_test_one_hot[i][class_to_group_mapping.get(y)] = 1

if use_class_and_group:
    output_sizes = [10, num_groups]
    one_hot_output = [y_train_one_hot, group_train_one_hot]  # to be removed
    output = [y_test, group_test]
    train_dataset = Dataset(x_train, x_train_noisy, y_train, group_train)

    training_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - training_size
    train_dataset, val_dataset = data.random_split(
        train_dataset, (training_size, val_size)
    )
    test_dataset = Dataset(x_test, x_test_noisy, y_test, group_test)

elif class_only:
    output_sizes = [10]
    one_hot_output = [y_train_one_hot]
    output = [y_test]
elif group_only:
    output_sizes = [num_groups]
    one_hot_output = [group_train_one_hot]
    output = [group_test]

train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False)
test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False)

classification_weights = (
    [10] if not use_class_and_group else [10, 10]
)  # Mess with these weights as desired.
proto_dist_weights = (
    [1] if not use_class_and_group else [1, 1]
)  # How realistic are the prototypes
feature_dist_weights = (
    [1] if not use_class_and_group else [1, 1]
)  # How close to prototypes are embeddings (cluster size)
disentangle_weights = [[0 for _ in range(2)] for _ in range(2)]
disentangle_weights[0] = [0, -10]
kl_losses = [1] if not use_class_and_group else [10, 10]
duplication_factors = [1] if not use_class_and_group else [1, 1]

all_seeds_hier_accs = [[] for i in range(len(output_sizes))]
all_ac = []
all_ed = []

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
        align_fn=torch.mean,
    )
    proto_model = proto_model.to(device)
    train(proto_model, 20, train_dataloader, val_dataloader)

    tmp_acc_lists, tmp_average_cost = eval_label_parity(
        proto_model, test_dataloader, gold_tree=(ground_truth_tree, class_labels)
    )
    for i, cur_seed_hier_accs in enumerate(all_seeds_hier_accs):
        cur_seed_hier_accs.append(tmp_acc_lists[i])

    tree = get_mst(
        proto_model, plot=False, labels=[class_labels, ["shoes", "top", "fancy"]]
    )
    tree_matches = trees_match(tree, ground_truth_tree)
    print("Trees match", tree_matches)
    # Super fast if the trees are actually close together.
    edit_dist = graph_edit_dist(ground_truth_tree, tree)
    # edit_dist = 0
    print("Edit distance", edit_dist)
    all_ac.append(tmp_average_cost)
    all_ed.append(edit_dist)

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

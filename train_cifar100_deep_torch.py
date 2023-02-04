import numpy as np
import tensorflow as tf
import torch
import torch.utils.data as data
from our_code.data_parsing.cifar_data import get_deep_data, get_deep_tree
from torchvision import transforms
from our_code.utils.metrics import trees_match, graph_edit_dist, get_deep_mst
from our_code.data_parsing.datasets import Deep_Cifar_Dataset
import torch.utils.data as data
from our_code.utils.train import train_Deep_Cifar
from our_code.utils.eval import eval_Deep_Cifar
from our_code.utils.metrics import get_mst
from our_code.models.proto_model import ProtoModel
from our_code.utils.seeds import set_seed


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Set seed before getting the data.
np.random.seed(0)
tf.random.set_seed(0)

latent_dim = 100
(
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
    all_labels,
) = get_deep_data()
# same method as before so no error here
_, _, ground_truth_tree = get_deep_tree()

output_sizes = [100, 20, 8, 4, 2]
one_hot_output = [
    y_train0_one_hot,
    y_train1_one_hot,
    y_train2_one_hot,
    y_train3_one_hot,
    y_train4_one_hot,
]
test_output = [y_test0, y_test1, y_test2, y_test3, y_test4]
#
classification_weights = [1 for _ in output_sizes]
classification_weights[0] = 5
proto_dist_weights = [0 for _ in output_sizes]
proto_dist_weights[0] = 0.1
feature_dist_weights = [0.1 for _ in output_sizes]
disentangle_weights = [[0 for _ in output_sizes] for _ in output_sizes]
disentangle_weights[0] = [0, -1, -1, -1, -1]
disentangle_weights[1] = [0, 0, -1, -1, -1]
disentangle_weights[2] = [0, 0, 0, -1, -1]
disentangle_weights[3] = [0, 0, 0, 0, -1]
kl_losses = [0 for _ in output_sizes]
duplication_factors = [1 for _ in output_sizes]

y_train0, y_train1, y_train2, y_train3, y_train4 = (
    np.squeeze(y_train0),
    np.squeeze(y_train1),
    np.squeeze(y_train2),
    np.squeeze(y_train3),
    np.squeeze(y_train4),
)
y_train0, y_train1, y_train2, y_train3, y_train4 = (
    y_train0.astype("int64"),
    y_train1.astype("int64"),
    y_train2.astype("int64"),
    y_train3.astype("int64"),
    y_train4.astype("int64"),
)

y_test0, y_test1, y_test2, y_test3, y_test4 = (
    np.squeeze(y_test0),
    np.squeeze(y_test1),
    np.squeeze(y_test2),
    np.squeeze(y_test3),
    np.squeeze(y_test4),
)
y_test0, y_test1, y_test2, y_test3, y_test4 = (
    y_test0.astype("int64"),
    y_test1.astype("int64"),
    y_test2.astype("int64"),
    y_test3.astype("int64"),
    y_test4.astype("int64"),
)

data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = Deep_Cifar_Dataset(
    x_train, x_train, y_train0, y_train1, y_train2, y_train3, y_train4, data_transforms
)

test_dataset = Deep_Cifar_Dataset(
    x_test, x_test, y_test0, y_test1, y_test2, y_test3, y_test4, data_transforms
)

training_size = int(len(train_dataset) * 0.8)
val_size = len(train_dataset) - training_size
train_dataset, val_dataset = data.random_split(train_dataset, (training_size, val_size))


train_dataloader = data.DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=2, shuffle=False)
test_dataloader = data.DataLoader(test_dataset, batch_size=2, shuffle=False)

all_ac = []
all_ed = []
all_seeds_hier_accs = [[] for i in range(len(output_sizes))]
# Run a bunch of trials.
for model_id in range(0, 2):
    set_seed(model_id)
    # Create, train, and eval the model
    proto_model = ProtoModel(
        output_sizes,
        decode_weight=0.1,
        duplication_factors=duplication_factors,
        input_size=32 * 32 * 3,
        classification_weights=classification_weights,
        proto_dist_weights=proto_dist_weights,
        feature_dist_weights=feature_dist_weights,
        disentangle_weights=disentangle_weights,
        kl_losses=kl_losses,
        latent_dim=latent_dim,
        in_plane_clusters=True,
        network_type="resnet",
        align_fn=torch.mean,
    )
    proto_model = proto_model.to(device)
    
    train_Deep_Cifar(
        proto_model, 60, train_dataloader, val_dataloader, all_labels, ground_truth_tree
    )
    mst = get_deep_mst(proto_model, add_origin=True,plot=False, labels=all_labels)

    tree_matches = trees_match(mst, ground_truth_tree)
    print("Tree matches", tree_matches)
    edit_dist = graph_edit_dist(ground_truth_tree, mst)
    print("Edit distance", edit_dist)
    tmp_acc_lists, tmp_average_cost = eval_Deep_Cifar(
        proto_model, test_dataloader, gold_tree=(ground_truth_tree, all_labels[0])
    )

    for i, cur_seed_hier_accs in enumerate(all_seeds_hier_accs):
        cur_seed_hier_accs.append(tmp_acc_lists[i])

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

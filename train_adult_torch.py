import numpy as np
import torch
import torch.utils.data as data
from our_code.data_parsing.adult_data import get_adult_data
from our_code.utils.train import train
from our_code.utils.eval import eval_fair
from our_code.models.proto_model import ProtoModel

from our_code.data_parsing.datasets import Dataset
from our_code.utils.seeds import set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(torch.cuda.is_available())

batch_size = 128
num_epochs = 20

y_accuracy = []
s_accuracy = []
disparate_impacts = []
demographics = []

for model_id in range(20):
    set_seed(model_id)

    (
        train_data,
        train_labels,
        train_protected,
        test_data,
        test_labels,
        test_protected,
    ) = get_adult_data("./data/adult.data", "./data/adult.test", wass_setup=False)
    input_size = train_data.shape[1]
    protected_shape = train_protected.shape
    protected_size = 1 if len(protected_shape) == 1 else train_protected.shape[1]

    # Create one-hot encodings of data
    train_labels_one_hot = train_labels
    train_protected_one_hot = train_protected
    train_protected = (
        train_protected_one_hot
        if protected_size == 1
        else np.argmax(train_protected_one_hot, axis=1)
    )
    test_labels_one_hot = test_labels
    test_labels = np.argmax(test_labels_one_hot, axis=1)
    test_protected_one_hot = test_protected
    test_protected = (
        test_protected_one_hot
        if protected_size == 1
        else np.argmax(test_protected_one_hot, axis=1)
    )

    protected_size = 2 
    output_sizes = [2, protected_size]
    train_outputs_one_hot = [train_labels_one_hot, train_protected_one_hot]
    test_outputs = [test_labels, test_protected]
    test_outputs_one_hot = [test_labels_one_hot, test_protected_one_hot]

    classification_weight = [1, 0.1]
    proto_dist_weights = [1, 1]
    feature_dist_weights = [1, 1]
    disentangle_weights = [[0, 100], [0, 0]]
    kl_losses = [0.1, 0.1]

    train_labels = train_labels[:, 1]

    train_dataset = Dataset(train_data, train_data, train_labels, train_protected)
    training_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - training_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, (training_size, val_size)
    )

    test_dataset = Dataset(test_data, test_data, test_labels, test_protected)

    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False)

    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False)

 
    proto_model = ProtoModel(
        output_sizes,
        input_size=input_size,
        decode_weight=0,
        classification_weights=classification_weight,
        proto_dist_weights=proto_dist_weights,
        feature_dist_weights=feature_dist_weights,
        disentangle_weights=disentangle_weights,
        kl_losses=kl_losses,
        latent_dim=32,
    )

    proto_model = proto_model.to(device)
    train(proto_model, num_epochs, train_dataloader, val_dataloader)

    y_accs, s_acc, s_diff, disparate_impact, demographic, slope = eval_fair(
        proto_model, test_dataloader, protected_idx=1
    )
    y_acc = y_accs[0]
    y_accuracy.append(y_acc)
    s_accuracy.append(s_acc)
    disparate_impacts.append(disparate_impact)
    demographics.append(demographic)
    # slopes.append(slope)
    # proto_model.viz_latent_space(test_data, test_labels, ['Low Income', 'High Income'])
    # proto_model.viz_latent_space(test_data, test_protected, [i for i in range(protected_size)], proto_indices=1)
    # proto_model.viz_projected_latent_space(test_data, test_labels, [i for i in range(2)], proto_indices=0)

print(
    "For all the seeds: The Y accuracy is", np.mean(y_accuracy), "±", np.std(y_accuracy)
)
print(
    "For all the seeds: The S accuracy is", np.mean(s_accuracy), "±", np.std(s_accuracy)
)
print(
    "For all the seeds: The Disparate Impact is ",
    np.mean(disparate_impacts),
    "±",
    np.std(disparate_impacts),
)
print(
    "For all the seeds: The Demographic is ",
    np.mean(demographics),
    "±",
    np.std(demographics),
)


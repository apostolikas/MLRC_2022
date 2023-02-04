import numpy as np
import torch.utils.data as data
from our_code.data_parsing.german_data import get_german_data
from our_code.utils.train import train
from our_code.utils.eval import eval_fair
from our_code.models.proto_model import ProtoModel
from our_code.utils.seeds import set_seed
from our_code.data_parsing.datasets import Dataset
import torch

def to_categorical(y, num_classes=None, dtype="float32"):
    """
    Function for replacing keras.to_caregorical.
    """
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
print(torch.cuda.is_available())

# Disentangle + KL need to be changed in order to reproduce Table 6
num_epochs = 30
classification_weight = [1, 1]
proto_dist_weights = [1, 1]
feature_dist_weights = [1, 1]
disentangle_weights = [[0, 100], [0, 0]]
# disentangle_weights = [[0, 0], [0, 0]]
# kl_losses = [0, 0]
kl_losses = [0.5, 0.5]
batch_size = 32


y_accuracy = []
s_accuracy = []
slopes = []
disentangle = True
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
    ) = get_german_data("./data/german_credit_data.csv", wass_setup=False)
    input_size = train_data.shape[1]

    train_labels_one_hot = to_categorical(train_labels, num_classes=2)
    train_protected_one_hot = to_categorical(train_protected)
    test_labels_one_hot = to_categorical(test_labels, num_classes=2)
    test_protected_one_hot = to_categorical(test_protected)

    train_dataset = Dataset(train_data, train_data, train_labels, train_protected)
    training_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - training_size
    train_dataset, val_dataset = data.random_split(
        train_dataset, (training_size, val_size)
    )
    test_dataset = Dataset(test_data, test_data, test_labels, test_protected)

    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False)
    test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False)

    num_protected_classes = train_protected_one_hot.shape[1]
    if disentangle:
        output_sizes = [2, num_protected_classes]
        train_outputs_one_hot = [train_labels_one_hot, train_protected_one_hot]
        test_outputs = [test_labels, test_protected]
        test_outputs_one_hot = [test_labels_one_hot, test_protected_one_hot]
    else:
        output_sizes = [2]  # Binary choice
        train_outputs_one_hot = [train_labels_one_hot]
        test_outputs = [test_labels]
        test_outputs_one_hot = [test_labels_one_hot]

    mean_train_labels = np.mean(train_labels)
    print("Mean test rate", mean_train_labels)
    mean_test_rate = np.mean(test_labels)
    print("Mean test rate", mean_test_rate)

    proto_model = ProtoModel(
        output_sizes,
        input_size=input_size,
        decode_weight=0,
        classification_weights=classification_weight,
        proto_dist_weights=proto_dist_weights,
        feature_dist_weights=feature_dist_weights,
        disentangle_weights=disentangle_weights,
        kl_losses=kl_losses,
    )
    proto_model = proto_model.to(device)

    train(
        proto_model,
        epochs=num_epochs,
        trainloader=train_dataloader,
        val_loader=val_dataloader,
    )
    
    y_accs, s_acc, s_diff, disparate_impact, demographic, slope = eval_fair(
        proto_model, test_dataloader, protected_idx=1
    )
    y_acc = y_accs[0]
    y_accuracy.append(y_acc)
    s_accuracy.append(s_acc)
    disparate_impacts.append(disparate_impact)
    demographics.append(demographic)
    slopes.append(slope)
    # proto_model.viz_latent_space(test_data, test_labels, ['Good Credit', 'Bad Credit'])
    # proto_model.viz_latent_space(test_data, test_protected, [i for i in range(2)], proto_indices=1)
    
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
print("For all the seeds: The ratio is ", np.mean(slopes), "±", np.std(slopes))

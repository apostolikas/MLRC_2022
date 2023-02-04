import torch
import numpy as np
import torch.nn as nn
from our_code.utils.metrics import (
    get_confusion_matrix,
    predict_protected,
    get_grad_orthogonality,
    get_disparity_metric,
)

# import seaborn as sn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def eval_label_parity(model, trainloader, gold_tree=None):
    """
    Evaluation function used for mnist digit and mnist fashion
    
    Params:
        model: proto_model, the model used
        trainloader: data.DataLoader, the training dataloader
        gold_tree: the correct tree

    Return:
        acc_lists: list, accuracies at every hierarchy level
        average_cost: int, the average cost 
    """
    loss_per_epoch = []
    epoch_loss = 0
    model.eval()
    mse = (
        nn.MSELoss()
    )  # better this way than mean and square despite the fact that they give the same results
    numerical_predictions = [[] for i in range(len(model.output_sizes))]
    all_ys = [[] for i in range(len(model.output_sizes))]
    # list with 2 arguments because we have y, parity_y for other cases different!
    num_evaluated = [0, 0]
    num_correct = [0, 0]
    square_loss = 0
    count_data = 0
    with torch.no_grad():
        for x, noisy_x, y, parity_y in trainloader:
            x = x.to(device)
            noisy_x = noisy_x.to(device)
            y = y.to(device)
            parity_y = parity_y.to(device)
            (
                dists_to_protos,
                mean_dists_to_protos,
                dists_to_latents,
                classification_preds,
                diffs_to_protos,
                mean_diffs_to_protos,
                proto_set_alignment,
                feature_enc_log_var,
                feature_enc,
            ) = model(
                noisy_x
            )  # it's a noisy autoencoder
            # classification_preds list of length 2 every element in the list batch_size, labels
            # eval_recons method in keras implementation
            recons = model.decoder(feature_enc)
            square_loss += torch.sum(torch.square(recons - x)).item()
            count_data += x.shape[0]
            all_ys[0].extend(y.tolist())
            all_ys[1].extend(parity_y.tolist())

            ys = (y, parity_y)
            # for every classification layer
            for i, test_prediction in enumerate(classification_preds):
                # if masks is not None:
                #     test_prediction = np.multiply(test_prediction, masks[predictor_idx][i])
                class_prediction = torch.argmax(
                    test_prediction, dim=-1
                )  # now dim=batch_size
                numerical_predictions[i].extend(class_prediction.tolist())
                num_evaluated[i] += class_prediction.shape[0]
                num_correct[i] += (class_prediction == ys[i]).sum().item()

    all_ys = [np.array(el) for el in all_ys]

    average_cost = get_confusion_matrix(
        model.output_sizes, all_ys, numerical_predictions, gold_tree
    )
    print("Average cost ", average_cost)

    reconstruction_loss = square_loss / count_data
    print("Mean reconstruction loss ", reconstruction_loss)
    acc_lists = []
    for count_eval, count_correct in zip(num_evaluated, num_correct):
        acc = count_correct / count_eval
        print("Accuracy is ", acc)
        acc_lists.append(acc)
    return acc_lists, average_cost


def eval_Deep_Cifar(model, trainloader, gold_tree=None):
    '''
    Evaluation method for the Cifar dataset having 5 levels of hierarchy
    
    Params:
        model: proto_model, the model used
        trainloader: data.DataLoader, the dataloader used for evaluation
        gold_tree: the correct tree
        
    Return:
        acc_lists: list, accuracies at every level
        average_cost: int, the average cost
    '''
    loss_per_epoch = []
    epoch_loss = 0
    model.eval()
    mse = (
        nn.MSELoss()
    )  # better this way than mean and square despite the fact that they give the same results
    numerical_predictions = [[] for i in range(len(model.output_sizes))]
    all_ys = [[] for i in range(len(model.output_sizes))]
    # list with 2 arguments because we have y, parity_y for other cases different!
    num_evaluated = [0 for i in range(len(model.output_sizes))]
    num_correct = [0 for i in range(len(model.output_sizes))]
    square_loss = 0
    count_data = 0
    with torch.no_grad():
        for x, noisy_x, y0, y1, y2, y3, y4 in trainloader:
            x = x.to(device)
            noisy_x = noisy_x.to(device)
            y0, y1, y2, y3, y4 = (
                y0.to(device),
                y1.to(device),
                y2.to(device),
                y3.to(device),
                y4.to(device),
            )
            (
                dists_to_protos,
                mean_dists_to_protos,
                dists_to_latents,
                classification_preds,
                diffs_to_protos,
                mean_diffs_to_protos,
                proto_set_alignment,
                feature_enc_log_var,
                feature_enc,
            ) = model(
                noisy_x
            )  # it's a noisy autoencoder
         
            count_data += x.shape[0]
            all_ys[0].extend(y0.tolist())
            all_ys[1].extend(y1.tolist())
            all_ys[2].extend(y2.tolist())
            all_ys[3].extend(y3.tolist())
            all_ys[4].extend(y4.tolist())

            ys = (y0, y1, y2, y3, y4)
            # for every classification layer
            for i, test_prediction in enumerate(classification_preds):
                class_prediction = torch.argmax(
                    test_prediction, dim=-1
                ) 
                numerical_predictions[i].extend(class_prediction.tolist())
                num_evaluated[i] += class_prediction.shape[0]
                num_correct[i] += (class_prediction == ys[i]).sum().item()

    all_ys = [np.array(el) for el in all_ys]

    average_cost = get_confusion_matrix(
        model.output_sizes, all_ys, numerical_predictions, gold_tree
    )
    print("Average cost ", average_cost)

    acc_lists = []
    for count_eval, count_correct in zip(num_evaluated, num_correct):
        acc = count_correct / count_eval
        print("Accuracy is ", acc)
        acc_lists.append(acc)
    return acc_lists, average_cost


def eval_fair(model, trainloader, protected_idx=1):
    '''
    Evaluation method for the fairness datasets

    Params:
        model: proto_model, the model we used
        trainloader: data.DataLoader, the training dataloader
        protected_idx: id of the protected input field
    Return:
        acc_lists: list, accuracy Y
        s_acc: float, s-accuracy of the protected field
        s_diff: float, difference between s_acc and random_chance
        disparate_impact: float, the disparate impact
        demographic: float, the demographic disparity (D.D-0.5)
        slopes: list of float, the rho metric
    '''
    numerical_predictions = [[] for i in range(len(model.output_sizes))]
    all_ys = [[] for i in range(len(model.output_sizes))]
    num_evaluated = [0, 0]
    num_correct = [0, 0]
    square_loss = 0
    count_data = 0
    all_feature_enc = []
    all_diffs_to_protos = []
    model.eval()
    with torch.no_grad():
        for x, noisy_x, y, parity_y in trainloader:
            x = x.to(device)
            noisy_x = noisy_x.to(device)
            y = y.to(device)
            parity_y = parity_y.to(device)
            (
                dists_to_protos,
                mean_dists_to_protos,
                dists_to_latents,
                classification_preds,
                diffs_to_protos,
                mean_diffs_to_protos,
                proto_set_alignment,
                feature_enc_log_var,
                feature_enc,
            ) = model(
                noisy_x
            )  # it's a noisy autoencoder
            recons = model.decoder(feature_enc)
            all_feature_enc.append(feature_enc)
            all_diffs_to_protos.append(diffs_to_protos[0])

            square_loss += torch.sum(torch.square(recons - x)).item()
            count_data += x.shape[0]
            all_ys[0].extend(y.tolist())
            all_ys[1].extend(parity_y.tolist())

            ys = (y, parity_y)

            for i, test_prediction in enumerate(classification_preds):

                class_prediction = torch.argmax(
                    test_prediction, dim=-1
                )  # now dim=batch_size
                numerical_predictions[i].extend(class_prediction.tolist())
                num_evaluated[i] += class_prediction.shape[0]
                num_correct[i] += (class_prediction == ys[i]).sum().item()

    all_ys = [np.array(el) for el in all_ys]
    all_feature_enc = torch.vstack(all_feature_enc)
    all_diffs_to_protos = torch.cat(all_diffs_to_protos, dim=0)

    _, _, slopes = get_grad_orthogonality(trainloader, model)

    disparate_impact, demographic, extreme_keys = get_disparity_metric(
        numerical_predictions, all_ys, protected_idx=protected_idx
    )

    s_acc, s_diff = predict_protected(
        all_diffs_to_protos,
        all_feature_enc,
        all_ys,
        protected_idx=protected_idx,
        keys=extreme_keys,
    )

    reconstruction_loss = square_loss / count_data
    print("Mean reconstruction loss ", reconstruction_loss)
    acc_lists = []

    for count_eval, count_correct in zip(num_evaluated, num_correct):
        acc = count_correct / count_eval
        print("Accuracy is ", acc)
        acc_lists.append(acc)

    return acc_lists, s_acc, s_diff, disparate_impact, demographic, slopes


def validate(model, valloader):
    '''
    validation method used for early stopping
    Params:
    '''
    model.eval()
    softmax = nn.Softmax(dim=-1)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    evaluation_loss = 0

    for x, noisy_x, y, parity_y in valloader:
        x = x.to(device)
        noisy_x = noisy_x.to(device)
        y = y.to(device)
        parity_y = parity_y.to(device)

        label_layers = (y, parity_y)

        (
            dists_to_protos,
            mean_dists_to_protos,
            dists_to_latents,
            classification_preds,
            _,
            mean_diffs_to_protos,
            proto_set_alignment,
            feature_enc_log_var,
            feature_enc,
        ) = model(noisy_x)
        recons = model.decoder(feature_enc)
        decode_loss = mse(recons, x)
        pred_loss = 0
        proto_dist_loss = 0
        feature_dist_loss = 0
        total_kl_loss = 0
        for i, label_layer in enumerate(label_layers):
            pred_loss_fn = ce(classification_preds[i], label_layer)
            pred_loss += model.classification_weights[i] * torch.mean(pred_loss_fn)

            min_proto_dist = torch.min(dists_to_protos[i], axis=-1).values  # returns 1d
            min_feature_dist = torch.min(
                dists_to_latents[i], axis=-1
            ).values  # returns 1d
            proto_dist_loss += model.proto_dist_weights[i] * torch.mean(
                min_proto_dist
            )  # R2 loss
            feature_dist_loss += model.feature_dist_weights[i] * torch.mean(
                min_feature_dist
            )  # R1 loss
            temperature = (
                0.1  # Smaller means more confident which one we're talking about.
            )
            softmaxed = softmax(-1 * (1.0 / temperature) * mean_dists_to_protos[i])
            reshaped = torch.reshape(
                softmaxed, shape=(-1, 1, mean_dists_to_protos[i].shape[1])
            )
            duplicated = torch.tile(reshaped, dims=[1, model.latent_dim, 1])
            squared_diffs = torch.square(mean_diffs_to_protos[i])
            product_loss = torch.multiply(duplicated, squared_diffs)
            dotted_loss = torch.sum(product_loss, axis=-1)
            mean_loss_term = dotted_loss
            kl_losses = (
                1
                + feature_enc_log_var
                - mean_loss_term
                - torch.exp(feature_enc_log_var)
            )
            kl_losses *= -0.5
            total_kl_loss += model.kl_losses[i] * torch.mean(kl_losses)

        alignment_loss = 0
        for i, alignment in enumerate(proto_set_alignment):
            for j, align in enumerate(alignment):
                alignment_loss += model.disentangle_weights[i][j] * align
        total_loss = (
            pred_loss
            + model.decode_weight * decode_loss
            + proto_dist_loss
            + feature_dist_loss
            + alignment_loss
            + total_kl_loss
        )
        evaluation_loss += total_loss.item()

    evaluation_loss = evaluation_loss / len(valloader.dataset)

    return evaluation_loss


def validate_Deep_CIFAR(model, valloader):
    model.eval()
    softmax = nn.Softmax(dim=-1)
    mse = nn.MSELoss()
    evaluation_loss = 0

    numerical_predictions = [[] for i in range(len(model.output_sizes))]

    num_evaluated = [0 for i in range(len(model.output_sizes))]
    num_correct = [0 for i in range(len(model.output_sizes))]
    cross_entropy = nn.CrossEntropyLoss()
    for x, noisy_x, y0, y1, y2, y3, y4 in valloader:
        x = x.to(device)
        noisy_x = noisy_x.to(device)
        y0, y1, y2, y3, y4 = (
            y0.to(device),
            y1.to(device),
            y2.to(device),
            y3.to(device),
            y4.to(device),
        )

        label_layers = (y0, y1, y2, y3, y4)

        (
            dists_to_protos,
            mean_dists_to_protos,
            dists_to_latents,
            classification_preds,
            _,
            mean_diffs_to_protos,
            proto_set_alignment,
            feature_enc_log_var,
            feature_enc,
        ) = model(noisy_x)

        pred_loss = 0
        proto_dist_loss = 0
        feature_dist_loss = 0
        total_kl_loss = 0
        for i, label_layer in enumerate(label_layers):
            pred_loss_fn = cross_entropy(classification_preds[i],label_layer)
            pred_loss += model.classification_weights[i] * torch.mean(pred_loss_fn)

            min_proto_dist = torch.min(dists_to_protos[i], axis=-1).values  # returns 1d
            min_feature_dist = torch.min(
                dists_to_latents[i], axis=-1
            ).values  # returns 1d
            proto_dist_loss += model.proto_dist_weights[i] * torch.mean(
                min_proto_dist
            )  # R2 loss
            feature_dist_loss += model.feature_dist_weights[i] * torch.mean(
                min_feature_dist
            )  # R1 loss
            temperature = (
                0.1  # Smaller means more confident which one we're talking about.
            )
            softmaxed = softmax(-1 * (1.0 / temperature) * mean_dists_to_protos[i])
            reshaped = torch.reshape(
                softmaxed, shape=(-1, 1, mean_dists_to_protos[i].shape[1])
            )
            duplicated = torch.tile(reshaped, dims=[1, model.latent_dim, 1])
            squared_diffs = torch.square(mean_diffs_to_protos[i])
            product_loss = torch.multiply(duplicated, squared_diffs)
            dotted_loss = torch.sum(product_loss, axis=-1)
            mean_loss_term = dotted_loss
            kl_losses = (
                1
                + feature_enc_log_var
                - mean_loss_term
                - torch.exp(feature_enc_log_var)
            )
            kl_losses *= -0.5
            total_kl_loss += model.kl_losses[i] * torch.mean(kl_losses)

            class_prediction = torch.argmax(
                classification_preds[i], dim=-1
            )  # now dim=batch_size
            numerical_predictions[i].extend(class_prediction.tolist())
            num_evaluated[i] += class_prediction.shape[0]
            num_correct[i] += (class_prediction == label_layers[i]).sum().item()

        alignment_loss = 0
        for i, alignment in enumerate(proto_set_alignment):
            for j, align in enumerate(alignment):
                alignment_loss += model.disentangle_weights[i][j] * align
        total_loss = (
            pred_loss
            + proto_dist_loss
            + feature_dist_loss
            + alignment_loss
            + total_kl_loss
        )
        evaluation_loss += total_loss.item()

    evaluation_loss = evaluation_loss / len(valloader.dataset)
    acc_lists = []
    i = 0
    for count_eval, count_correct in zip(num_evaluated, num_correct):
        acc = count_correct / count_eval
        print("Accuracy is ", acc, " level:", i)
        acc_lists.append(acc)
        i += 1

    return evaluation_loss, acc_lists

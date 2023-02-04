import torch
import torch.nn as nn
from our_code.utils.metrics import get_mst
from our_code.utils.metrics import graph_edit_dist
from our_code.utils.eval import validate, validate_Deep_CIFAR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def train(model, epochs, trainloader, val_loader=None):
    # optimizers and params
    optimizer = torch.optim.Adam(model.parameters())
    mse = (
        nn.MSELoss()
    )  # better this way than mean and square despite the fact that they give the same results
    softmax = nn.Softmax(dim=-1)
    ce = nn.CrossEntropyLoss()
    loss_per_epoch = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.01,
        patience=3,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-6,
        eps=1e-08,
        verbose=True,
    )

    counter = 0
    min_delta = 0
    tolerance = 10

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
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
                _,
                mean_diffs_to_protos,
                proto_set_alignment,
                feature_enc_log_var,
                feature_enc,
            ) = model(
                noisy_x
            )  # it's a noisy autoencoder
            recons = model.decoder(feature_enc)
            decode_loss = mse(recons, x)

            label_layers = (y, parity_y)
            # init losses
            pred_loss = 0
            proto_dist_loss = 0
            feature_dist_loss = 0
            total_kl_loss = 0
            for i, label_layer in enumerate(label_layers):
                pred_loss_fn = ce(classification_preds[i], label_layer)
                pred_loss += model.classification_weights[i] * torch.mean(pred_loss_fn)

                # losses from the other paper
                min_proto_dist = torch.min(
                    dists_to_protos[i], axis=-1
                ).values  # returns 1d
                min_feature_dist = torch.min(
                    dists_to_latents[i], axis=-1
                ).values  # returns 1d
                proto_dist_loss += model.proto_dist_weights[i] * torch.mean(
                    min_proto_dist
                )  # R2 loss
                feature_dist_loss += model.feature_dist_weights[i] * torch.mean(
                    min_feature_dist
                )  # R1 loss

                # kl divergence
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

            #  alignment loss
            alignment_loss = 0
            for i, alignment in enumerate(proto_set_alignment):
                for j, align in enumerate(alignment):
                    alignment_loss += model.disentangle_weights[i][j] * align

            big_loss = (
                pred_loss
                + model.decode_weight * decode_loss
                + proto_dist_loss
                + feature_dist_loss
                + alignment_loss
                + total_kl_loss
            )
            optimizer.zero_grad()
            big_loss.backward()  # retain_graph=True
            optimizer.step()

            epoch_loss += big_loss.item()
        epoch_loss /= len(trainloader)
        loss_per_epoch.append(epoch_loss)

        print("Epoch: ", epoch + 1, " training loss: ", epoch_loss)
        if val_loader is not None:
            val_loss = validate(model, val_loader)
            print("Epoch: ", epoch + 1, " val loss: ", val_loss)
            if (val_loss - epoch_loss) > min_delta:
                counter += 1
                if counter >= tolerance:
                    print("Early stopping stopped at epoch:", i)
                    break
            else:
                counter = 0
            scheduler.step(val_loss)


def train_Deep_Cifar(
    model, epochs, trainloader, val_loader, all_labels, ground_truth_tree
):
    # optimizers and params
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, nesterov=False, momentum=0.9
    )
    mse = (
        nn.MSELoss()
    )  # better this way than mean and square despite the fact that they give the same results
    softmax = nn.Softmax(dim=-1)
    cross_entropy = nn.CrossEntropyLoss()
    loss_per_epoch = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.01,
        patience=3,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-6,
        eps=1e-08,
        verbose=True,
    )

    counter = 0
    min_delta = 0
    tolerance = 10

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
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
                _,
                mean_diffs_to_protos,
                proto_set_alignment,
                feature_enc_log_var,
                feature_enc,
            ) = model(
                noisy_x
            )  # it's a noisy autoencoder

            label_layers = (y0, y1, y2, y3, y4)
            # init losses
            pred_loss = 0
            proto_dist_loss = 0
            feature_dist_loss = 0
            total_kl_loss = 0
            for i, label_layer in enumerate(label_layers):
                
                pred_loss_fn = cross_entropy(classification_preds[i],label_layer)
                pred_loss += model.classification_weights[i] * torch.mean(pred_loss_fn)

                # losses from the other paper
                min_proto_dist = torch.min(
                    dists_to_protos[i], axis=-1
                ).values  # returns 1d
                min_feature_dist = torch.min(
                    dists_to_latents[i], axis=-1
                ).values  # returns 1d
                proto_dist_loss += model.proto_dist_weights[i] * torch.mean(
                    min_proto_dist
                )  # R2 loss
                feature_dist_loss += model.feature_dist_weights[i] * torch.mean(
                    min_feature_dist
                )  # R1 loss

                # kl divergence
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

            #  alignment loss
            alignment_loss = 0
            for i, alignment in enumerate(proto_set_alignment):
                for j, align in enumerate(alignment):
                    alignment_loss += model.disentangle_weights[i][j] * align

            big_loss = (
                pred_loss
                
                + proto_dist_loss
                + feature_dist_loss
                + alignment_loss
                + total_kl_loss
            )
            optimizer.zero_grad()
            big_loss.backward()  
            optimizer.step()

            epoch_loss += big_loss.item()

        epoch_loss /= len(trainloader)
        loss_per_epoch.append(epoch_loss)

        print("Epoch: ", epoch + 1, " training loss: ", epoch_loss)
        if val_loader is not None:
            val_loss, acc_lists = validate_Deep_CIFAR(model, val_loader)
            print("Epoch: ", epoch + 1, " val loss: ", val_loss)
            if (val_loss - epoch_loss) > min_delta:
                counter += 1
                if counter >= tolerance:
                    print("Early stopping stopped at epoch:", i)
                    break
            else:
                counter = 0

            scheduler.step(val_loss)
            if epoch == 10 or (epoch > 10 and epoch % 4 == 0):
                mst = get_mst(model, add_origin=True, plot=False, labels=all_labels)
                edit_dist = graph_edit_dist(ground_truth_tree, mst)
                print("INSIDE TRAIN!! Edit distance", edit_dist)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss,
                "accuracies": acc_lists,
            },
            "DeepCifar.pt",
        )

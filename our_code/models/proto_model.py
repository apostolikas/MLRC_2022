import torch
import torch.nn as nn
import numpy as np

from our_code.models.proto_layer import ProtoLayer
from our_code.models.dist_prob_layer import DistProbLayer
from our_code.models.encoder import MLP_Encoder, Encoder, ResNet_Encoder
from our_code.models.decoder import MLP_Decoder,ConvDecoder

# ! only for pytorch this file!
class ProtoModel(nn.Module):
    def __init__(
        self,
        output_sizes,
        duplication_factors=None,
        input_size=784,
        decode_weight=1,
        classification_weights=None,
        proto_dist_weights=None,
        feature_dist_weights=None,
        disentangle_weights=None,
        kl_losses=None,
        latent_dim=32,
        proto_grids=None,
        in_plane_clusters=True,
        use_shadow_basis=False,
        align_fn=torch.max,
        network_type="dense",
    ):
        """
        Params:
            output_sizes: list containg the number of different labels in each classification problem
            duplication_factors: list containing the number of prototypes each label will have at each classification problem
            input_size: int representing the dimensionality of the input
            decode_weight: list of lamdas hyperparemeters used for the decoder
            classification_weights: list of lamdas hyperparameters for the cross entropy loss
            proto_dist_weights: list of lamdas hyperparameters for the proto_dist loss i.e. R2 loss in PCN paper
            feature_dist_weights: list of lamdas hyperparameters for the feature_dist loss i.e. R2 loss in PCN paper
            disentangle_weights: list of lamdas hyperparameters for the allignment loss
            kl_losses: list of lamdas hyperparameters for the kl-loss
            latent_dim: int representing the dimensionality of the latent space
            in_plane_clusters: boolean whether we want in_plane clusters
            use_shadow_basis: boolean whether we want to apply QR simultaneously to all the prototypes together
            align_fn: function that is applied to terms that are part of the allignment loss

        """
        super(ProtoModel, self).__init__()
        # super().__init__()
        self.decode_weight = decode_weight
        self.latent_dim = latent_dim
        self.output_sizes = output_sizes  # list where every element contains the number of labels at each level in the hierarchy
        self.in_plane_clusters = in_plane_clusters
        # ! be sure to have the same arguments with pytorch
        self.align_fn = align_fn
        if duplication_factors is None:
            self.duplication_factors = [1 for _ in range(len(output_sizes))]
        else:
            self.duplication_factors = duplication_factors
        self.classification_weights = []
        self.proto_dist_weights = []
        self.feature_dist_weights = []
        self.kl_losses = []
        # Should be a list of lists, where all lists have size n.
        if disentangle_weights is None or disentangle_weights == 0:
            disentangle_weights = []
            for _ in range(len(output_sizes)):
                disentangle_weights.append([0 for _ in range(len(output_sizes))])
        self.disentangle_weights = disentangle_weights
        for i, output_size in enumerate(output_sizes):
            if classification_weights is not None:
                self.classification_weights.append(classification_weights[i])
            else:
                self.classification_weights.append(1)
            if proto_dist_weights is not None:
                self.proto_dist_weights.append(proto_dist_weights[i])
            else:
                self.proto_dist_weights.append(1)
            if feature_dist_weights is not None:
                self.feature_dist_weights.append(feature_dist_weights[i])
            else:
                self.feature_dist_weights.append(1)
            if kl_losses is not None:
                self.kl_losses.append(kl_losses[i])
            else:
                self.kl_losses.append(0)
        self.input_size = input_size
        self.use_shadow_basis = use_shadow_basis

        # ! because we have an MLP we have to support other types as well
        self.create_proto_classifiers()
        if network_type == "dense_mnist" or network_type == "dense":
            self.build_network_parts()
        elif network_type == "resnet":
            self.build_conv_network_parts()

        # ! maybe not a list but numpy or torch?
        self.label_mask_layers = [
            torch.ones(output_size) for output_size in self.output_sizes
        ]

    def create_proto_classifiers(self):
        """
        create prototype layer and classifier layer
        """
        self.proto_layers = nn.ModuleList()
        self.classifier_layers = nn.ModuleList()

        for i, output_size in enumerate(self.output_sizes):
            self.proto_layers.append(
                ProtoLayer(
                    num_prototypes=output_size * self.duplication_factors[i],
                    dim=self.latent_dim,
                    in_plane=self.in_plane_clusters,
                )
            )
            if output_size > 1:
                self.classifier_layers.append(
                    DistProbLayer(
                        num_classes=output_size,
                        duplication_factor=self.duplication_factors[i],
                    )
                )
            else:
                raise ValueError("output_size <= 1")

    def build_network_parts(self):
        """
        create MLP encoder and MLP decoder model
        """
        encoder_mlp = MLP_Encoder(input_dim=self.input_size, hidden_dim=128)

        self.encoder = Encoder(encoder_model=encoder_mlp, latent_dim=self.latent_dim)

        self.decoder = MLP_Decoder(
            input_dim=self.latent_dim, hidden_dim=128, output_dim=self.input_size
        )

    def build_conv_network_parts(self):
        """
        create ResNet encoder for the CIFAR task
        """
        encoder_resnet = ResNet_Encoder(mlp_hidden_dim=4096)

        self.encoder = Encoder(encoder_model=encoder_resnet, latent_dim=self.latent_dim)
        self.decoder = ConvDecoder(
            latent_dim =self.latent_dim,
            input_size = self.input_size
        )

    def forward(self, x):
        """
        forward method of the whole architecture

        Params:
            x: input tensor the model
        Return:
            dists_to_protos: tensor, distance from the feature_enc (after the reparameterization_trick) to the prototypes
            mean_dists_to_protos: tensor, distance between the feature_enc_mean and the prototypes
            dists_to_latents: tensor, distance from the prototypes to the feature_enc
            classification_preds: tensor, predictions of the model before applying cross entropy
            diffs_to_protos: tensor, difference from the feature_enc to prototypes
            mean_diffs_to_protos: tensor, difference from the feature_enc_mean to prototypes
            proto_set_alignment: tensor, list containing the allignment terms that will be used for the allignment loss
            feature_enc_log_var: tensor, variance tensor from the encoder output
            feature_enc: tensor, output of the encoder
        """
        feature_enc_mean, feature_enc_log_var, feature_enc = self.encoder(x)

        self.shadow_basis = None
        if self.use_shadow_basis and len(self.proto_layers) > 1:
            all_other_protos = []
            for proto_layer in self.proto_layers:
                all_other_protos.extend(torch.unbind(proto_layer.prototypes))
            proto0 = all_other_protos[0]
            difference_vectors = []
            for i, proto in enumerate(all_other_protos):
                if i == 0:
                    continue
                difference_vectors.append(proto - proto0)
            diff_tensor = torch.stack(difference_vectors)
            Q, _ = torch.linalg.qr(torch.transpose(diff_tensor))
            self.shadow_basis = Q

        dists_to_protos = []
        mean_dists_to_protos = []
        dists_to_latents = []
        classification_preds = []
        diffs_to_protos = []
        mean_diffs_to_protos = []
        for i, proto_layer in enumerate(self.proto_layers):
            # put in proto_layer the z
            dist_to_protos, dist_to_latents, diff_to_protos = proto_layer(feature_enc)
            # put in proto_layer the mean
            mean_dist_to_protos, _, mean_diff_to_protos = proto_layer(feature_enc_mean)

            classification_pred = self.classifier_layers[i](
                [dist_to_protos, self.label_mask_layers[i]]
            )

            dists_to_protos.append(dist_to_protos)
            mean_dists_to_protos.append(mean_dist_to_protos)
            dists_to_latents.append(dist_to_latents)
            diffs_to_protos.append(diff_to_protos)
            mean_diffs_to_protos.append(mean_diff_to_protos)
            classification_preds.append(classification_pred)


        self.vector_diffs = []
        for set_idx, proto_set in enumerate(self.proto_layers):
            self.vector_diffs.append(proto_set.vector_diffs)
        if self.shadow_basis is not None:
            print("Adding shadow basis")
            self.vector_diffs.append(self.shadow_basis)

        proto_set_alignment = []
        for i, vector_set1 in enumerate(self.vector_diffs):
            alignment = []
            for j, vector_set2 in enumerate(self.vector_diffs):
                # cosine with other concept subspaces
                cosines = torch.matmul(torch.transpose(vector_set1, 0, -1), vector_set2)
                cos_squared = torch.pow(cosines, 2)
                alignment.append(self.align_fn(cos_squared))
            proto_set_alignment.append(alignment)

        return (
            dists_to_protos,
            mean_dists_to_protos,
            dists_to_latents,
            classification_preds,
            diffs_to_protos,
            mean_diffs_to_protos,
            proto_set_alignment,
            feature_enc_log_var,
            feature_enc,
        )  

    def get_prototypes(self):
        """
        return the prototypes
        Return:
            prototypes: list containing the prototypes
        """
        prototypes = []
        for proto_layer in self.proto_layers:
            prototypes.extend(proto_layer.prototypes.detach().cpu().numpy())
        return prototypes

    @staticmethod
    def get_vector_differences(protos):
        """
        Return:
            diffs: list of list containing the normalized differences between each pair of prototypes at each classification problem
        """
        diffs = []
        for i, proto1 in enumerate(protos):
            diffs_for_i = []
            for j, proto2 in enumerate(protos):
                if i == j:
                    diffs_for_i.append(0)
                    continue
                diffs_for_i.append((proto1 - proto2) / np.linalg.norm(proto1 - proto2))
            diffs.append(diffs_for_i)
        return diffs

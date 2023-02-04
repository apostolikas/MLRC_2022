import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ProtoLayer(nn.Module):
    '''
    The prototype layer of the CSN model
    '''
    def __init__(self, num_prototypes, dim, fixed_protos=None, in_plane=True):
        '''
        Params:
            num_prototypes: Integer, the number of total prototypes
            dim: Integer, the dimensionality of latent space
            fixed_protos: Iterable, the prototypes if they remain fixed
            in_plane: Boolean, True if in the plane
        '''
        super(ProtoLayer, self).__init__()
        self.num_prototypes = num_prototypes
        self.latent_dim = dim
        self.fixed_protos = fixed_protos
        self.in_plane = in_plane
        if self.fixed_protos is not None:
            print("Non trainable weights")
            self.prototypes = torch.randn(list(self.fixed_proto),requires_grad=False)

        else:
            print("Trainable weights")
            temp_prototypes = torch.rand((self.num_prototypes,self.latent_dim))
            self.prototypes = nn.Parameter(temp_prototypes, requires_grad=True)
            nn.init.uniform_(self.prototypes)

        self.vector_diffs = self.get_proto_basis()

    @staticmethod
    def get_norms(x):
        '''
        Params:
            x: tensor
        Return:
            the norm of the input tensor
        '''
        return torch.sum(torch.pow(x,2),axis=1)

    def forward(self, feature_vectors):
        '''
        Params:
            feature_vectors: tensor representing either the predicted mean of the latent distribution or the latent vectors
        Return:
            list containing distances between feature vectors and prototypes and between projected features and prototypes 
        '''
        self.vector_diffs = self.get_proto_basis()
        # The normal distance terms:
        # Compute the distance between x and the protos
        dists_to_protos = ProtoLayer.get_distances(feature_vectors, self.prototypes)
        dists_to_latents = ProtoLayer.get_distances(self.prototypes, feature_vectors) # NOT same as above

        # Calculate the projected feature vectors in the subspace defined by the prototypes
        projected_features = self.get_projection_in_subspace(feature_vectors)
        dists_in_plane_protos = ProtoLayer.get_distances(projected_features, self.prototypes)
        dists_in_plane_latents = ProtoLayer.get_distances(self.prototypes, projected_features) # NOT same as above

        # Compute the difference vector from each feature encoding to each prototype.
        diffs_to_protos = self.get_dist_to_protos(feature_vectors)
        # Or, here, compute the diffs to protos using only the components outside the plane.
        in_plane_diffs_to_protos = self.get_dist_to_protos(projected_features)
        out_of_plane_diffs = -1 * (diffs_to_protos - in_plane_diffs_to_protos)

        if not self.in_plane:
            return [dists_to_protos, dists_to_latents, diffs_to_protos]
        return [dists_in_plane_protos, dists_in_plane_latents, out_of_plane_diffs]

    @staticmethod
    def get_distances(tensor1, tensor2):
        '''
        Calculate the distance between 1st tensor and the 2nd tensor

        Params:
            tensor1: multidimensional tensor
            tensor2: multidimensional tensor  
        Return:
            dists_between: distance between tensor 1 and tensor 2
        '''
        t1_squared = torch.reshape(ProtoLayer.get_norms(tensor1), shape=(-1, 1))
        t2_squared = torch.reshape(ProtoLayer.get_norms(tensor2), shape=(1, -1))
   
        dists_between = t1_squared + t2_squared - 2 * torch.matmul(tensor1, torch.transpose(tensor2,0,-1))

        return dists_between

    def get_proto_basis(self):
        '''
        calculate an orthogonal basis using QR defined by the prototypes.

        Return:
            Q: tensor of shape [self.latent_dim, self.num_prototypes], the orthogonal basis of the concept subspace
        '''
        if self.latent_dim < self.num_prototypes - 1:
            print("Assuming that prototypes span the whole space, which isn't necessarily true.")
            np_array = np.zeros((self.latent_dim, self.num_prototypes - 1))
            for i in range(self.latent_dim):
                np_array[i, i] = 1

            return torch.tensor(np_array, requires_grad=False,dtype=torch.float32)

        unstacked_protos = torch.unbind(self.prototypes)#converts 2d to tuple 

        proto0 = self.prototypes[0]
        difference_vectors = []
        for i, proto in enumerate(unstacked_protos):
            if i == 0:
                continue
            difference_vectors.append(proto - proto0)
        difference_tensor = torch.stack(difference_vectors)#2d

        Q, R = torch.linalg.qr(torch.transpose(difference_tensor,0,-1))

        # Q has latent_dim rows and num_protos - 1 columns
        # Another version would be to use the prototypes as the basis, but that creates one extra dimension.
        return Q

    def get_projection_in_subspace(self, features):
        '''
        Calculate an offset to move into a space relative to the first prototype.
        This offset effect is later undone after projecting down.
        
        Params:
            features: tensor representing features
        Return:
            global_projected_features: tensor representing global projected features
        '''
        offset = torch.tile(torch.reshape(self.prototypes[0], (1, self.latent_dim)), [features.shape[0], 1])

        relative_features = features - offset
        feature_dotted = torch.matmul(relative_features, self.vector_diffs)
        projected_features = torch.matmul(feature_dotted, torch.transpose(self.vector_diffs,0,-1))
        global_projected_features = projected_features + offset
        return global_projected_features

    def get_dist_to_protos(self, feature_vectors):
        '''
        Compute the difference vector from each feature_vector to each prototype

        Params:
            feature_vectors: tensor representing features

        Return:
            diffs_to_protos: tensor representing the difference vector from each feature_vector to each prototype
        '''
        features_shaped = torch.reshape(feature_vectors, shape=(-1, self.latent_dim, 1))
        repeated_features = torch.tile(features_shaped, dims=[1, 1, self.num_prototypes])

        transposed_prototypes = torch.transpose(self.prototypes,0,-1)
        protos_shaped = torch.reshape(transposed_prototypes, shape=(1, self.latent_dim, self.num_prototypes))

        repeated_protos = torch.tile(protos_shaped, dims=[feature_vectors.shape[0], 1, 1])
        diffs_to_protos = repeated_protos - repeated_features

        return diffs_to_protos

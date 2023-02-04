import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class DistProbLayer(nn.Module):
    """
    This Layer is after the Proto Layer
    """

    def __init__(self, num_classes, duplication_factor):
        """
        Params:
            num_classes: Integer, the number of different classes for a classification problem
            duplication_factor: Integer, the number of different prototype vectors we have for each class
        """
        super(DistProbLayer, self).__init__()
        self.num_classes = num_classes
        self.duplication_factor = duplication_factor

    def forward(self, inputs):
        """
        Params:
            inputs: tuple containing tensor distances calculated in the Proto_Layer and the masks tensor for current classification problem
        Return:
            unnormalized_prob: tensor containing the unnormalized probabilities
        """
        distances, mask = inputs
        mask = mask.to(device)
        distances = distances.to(device)

        # get the negative distance
        unnormalized_prob = DistProbLayer.dist_to_prob(distances)
        unnormalized_prob = unnormalized_prob.to(device)

        return unnormalized_prob

    @staticmethod
    def dist_to_prob(dist):
        """
        return the negative of the input distance

        Params:
            dist: tensor representing some kind of distance

        Return:
            the negative distance
        """
        return -dist

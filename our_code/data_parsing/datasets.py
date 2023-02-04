import torch.utils.data as data
import numpy as np
from PIL import Image


# Subclass
class Dataset(data.Dataset):
    """
    General dataset class for most experiments.
    """

    def __init__(self, x, noisy_x, label, parity_label):
        """
        Params:
            x: data
            noisy_x: data with noise
            label: feature labels
            parity_label: protected feature
        """
        super().__init__()

        self.x = x
        self.noisy_x = noisy_x
        self.label = label
        self.parity_label = parity_label

    def __len__(self):
        """
        Number of data point we have.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Return the idx-th data point of the dataset and its labels.
        """
        data_point = self.x[idx]
        noisy_data_point = self.noisy_x[idx]
        data_label = self.label[idx]
        data_parity_label = self.parity_label[idx]
        return (
            data_point.astype(np.float32),
            noisy_data_point.astype(np.float32),
            data_label,
            data_parity_label,
        )


class Deep_Cifar_Dataset(data.Dataset):
    """
    Custom dataset class for CIFAR 100 with 5 levels of hierarchy.
    """

    def __init__(
        self,
        x_train,
        x_noisy,
        y_train0,
        y_train1,
        y_train2,
        y_train3,
        y_train4,
        transform,
    ):
        """
        Params:
            x_train: data
            x_noisy: data with noise
            y_train0, y_train1, y_train2, y_train3, y_train4: labels for each level of hierarchy
            transform: list of image transformations
        """
        super().__init__()

        self.x_train = x_train
        self.x_noisy = x_noisy

        self.y_train0 = y_train0
        self.y_train1 = y_train1
        self.y_train2 = y_train2
        self.y_train3 = y_train3
        self.y_train4 = y_train4

        self.transform = transform

    def __len__(self):
        """
        Number of data point we have.
        """
        return len(self.x_train)

    def __getitem__(self, idx):
        """
        Return the idx-th data point of the dataset.
        """

        x_train = self.x_train[idx]
        x_noisy = self.x_noisy[idx]

        x_train = Image.fromarray(x_train.astype("uint8"))
        x_noisy = Image.fromarray(x_noisy.astype("uint8"))

        x_train = self.transform(x_train)
        x_noisy = self.transform(x_noisy)

        y_train0 = self.y_train0[idx]
        y_train1 = self.y_train1[idx]
        y_train2 = self.y_train2[idx]
        y_train3 = self.y_train3[idx]
        y_train4 = self.y_train4[idx]

        return x_train, x_noisy, y_train0, y_train1, y_train2, y_train3, y_train4

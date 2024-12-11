from torch import flatten
import torch.nn as nn


class LeNet(nn.Module):
    """LeNet implementation designed for 28x28 greyscale images

    This model consists of 2 convolutional blocks, followed by a fully connected layer.
    The kernel for the convolutional layers is fixed to the size (5,5) whereas the pooling
    stride and kernel are set to be (2,2) respectively.

    The final layer consists of a logSoftmax activation.

    (Source: https://en.wikipedia.org/wiki/LeNet#:~:text=LeNet%20is%20a%20series%20of,%2D5%20architecture%20(overview).)

    Disclaimer: This network is not checked to align 100% with the works of LeCun et. al., it must be rather referred to
    as "LeNet-inspired" as LeNet-5 uses slightly more features in the FCN layer and has a fixed number of output classes.
    """

    def __init__(
        self,
        channels: int = 1,  # Number of channels at the input layer
        classes: int = 1,  # Number of possible classes at the output layer
    ):
        super(LeNet, self).__init__()

        self.kernel = (5, 5)
        self.pool_kernel = (2, 2)
        self.pool_stride = (2, 2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=20, kernel_size=self.kernel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=self.kernel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=800, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(
        self,
        x,
        return_embedding: bool = False,  # Set to True, if the CNN embeddings shall be the output along with the predicted class
    ):
        x_embedded = self.conv(x)
        x = self.fc(flatten(x_embedded, 1))

        return x if return_embedding is False else x, x_embedded

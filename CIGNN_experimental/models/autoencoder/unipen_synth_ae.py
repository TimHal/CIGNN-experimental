import torch.nn as nn


class UnipenSyntheticAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.kernel = (8, 8)
        self.stride = (2, 2)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, self.kernel, self.stride),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Conv2d(3, 5, self.kernel, self.stride),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Conv2d(5, 10, self.kernel, self.stride),
            nn.Dropout(p=0.2),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(10, 5, self.kernel, self.stride),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.ConvTranspose2d(5, 3, self.kernel, self.stride),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, self.kernel, self.stride),
            nn.Sigmoid(),
        )

    def forward(self, x):

        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        return x

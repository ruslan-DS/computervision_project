import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.SELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.SELU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.SELU()
        )

        #decoder
        self.conv1_t = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=2),
            nn.BatchNorm2d(64),
            nn.SELU()
        )
        self.conv2_t = nn.Sequential(
            nn.ConvTranspose2d(64, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.SELU()
        )

        self.conv3_t = nn.Sequential(
            nn.ConvTranspose2d(256, 1, kernel_size=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def decode(self, x):
        x = self.conv1_t(x)
        x = self.conv2_t(x)
        x = self.conv3_t(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        return out
    
device = 'cpu'    
model = ConvAutoencoder().to(device)
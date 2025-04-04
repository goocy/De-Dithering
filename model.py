import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedUpsampler(nn.Module):
    # FeatUp (https://arxiv.org/html/2403.10516v1)
    def __init__(self):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, padding=1)
        )
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        return torch.sigmoid(self.upscale(features))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        upscale_factor = 2
        n = upscale_factor ** 2

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128,
                               kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=n*3,
                               kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # Pass-through layers
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x1))
        x3 = F.leaky_relu(self.conv3(x2))
        x4 = F.leaky_relu(self.conv4(x3))
        x5 = F.leaky_relu(self.conv5(x4))
        x6 = self.pixel_shuffle(x5)

        # Final output
        out = torch.sigmoid(x6)
        return out
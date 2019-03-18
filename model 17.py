import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        upscale_factor = 1
        n = upscale_factor ** 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, n*3, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x)

        x = self.conv2(x)
        x = nn.functional.leaky_relu(x)

        x = self.conv3(x)
        x = nn.functional.leaky_relu(x)

        x = self.conv4(x)
        #x = self.pixel_shuffle(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    model = Net()
    print(model)

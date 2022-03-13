import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

# 搭建神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# if __name__ == '__main__':
#     cnn = CNN()
#     input = torch.ones((64, 3, 32, 32))
#     output = cnn(input)
#     print(output.shape)
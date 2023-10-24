import torch
import numpy as np
import torch.optim as optim
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F


# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# 定义ResNet
class ResNet18(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResNet18, self).__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self.make_layer(block, 64, 64, 2)
        self.layer2 = self.make_layer(block, 64, 128, 2)
        self.layer3 = self.make_layer(block, 128, 256, 2)
        self.layer4 = self.make_layer(block, 256, 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



# 定义训练函数
def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.cuda()
        target=target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


# 定义测试函数
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.cuda()
            target=target.cuda()

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


# 准备数据
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST("D:\\deep_learning\\datasets\\MNIST\\", train=True, download=True, transform=transform)
test_set = datasets.MNIST("D:\\deep_learning\\datasets\\MNIST\\", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)

img,label = train_set[21]
# plt.imshow(img.squeeze(), cmap='gray')
# plt.title('Label: %d' % label)
plt.axis('on')
plt.imshow(img.squeeze(),cmap='gray')
plt.show()
# 实例化模型
model = ResNet18(ResBlock)
model = model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
criterion=criterion.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10):
    train(model, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, test_loader, criterion)
    print('Epoch: {} \tTest Loss: {:.4f} \tTest Acc: {:.2f}%'.format(epoch, test_loss, test_acc))
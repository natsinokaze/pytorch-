import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", transform= torchvision.transforms.ToTensor(), train=False, download=True)

dataloader = DataLoader(dataset,batch_size=1)

class Seq(nn.Module):
    def __init__(self):
        super(Seq, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 3@32*32 --> 32@32*32
            MaxPool2d(2),  # 32@32*32 --> 32@16*16
            Conv2d(32, 32, 5, padding=2),  # 32@16*16 --> 32@16*16
            MaxPool2d(2),  # 32@16*16 --> 32@8*8
            Conv2d(32, 64, 5, padding=2),  # 32@8*8 --> 64@8*8
            MaxPool2d(2),  # 64@8*8 --> 64@4*4
            Flatten(),
            Linear(1024, 64),  # 第一项 一定要正确这里是1024的原因是经过maxpool3后有64*4*4=1024个元素
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

seq = Seq()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    outputs = seq(imgs) # 得到的结果为batch_size=1的10个类的线性值 注意不是概率
    #print(outputs) # 能输出结果则说明模型没有问题
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print(result_loss)

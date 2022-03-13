import torch
import torchvision
from  torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", transform= torchvision.transforms.ToTensor(), train=False, download=True)

dataloader = DataLoader(dataset,batch_size=64)

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x
conv = Conv()

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, target = data
    output = conv(imgs)# 输入一个参数 调用了forward方法

    # imgs.shape的结果为torch.Size([64, 3, 32, 32])
    # (output.shape的结果为torch.Size([64, 6, 30, 30])

    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1, 3, 30,30))#当输入-1时系统会自动根据情况调整batchsize
    writer.add_images("output", output, step)
    step = step + 1



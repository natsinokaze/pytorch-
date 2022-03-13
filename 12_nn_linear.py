import torch
import torchvision
from torch.nn import MaxPool2d, ReLU, Sigmoid, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", transform= torchvision.transforms.ToTensor(), train=False, download=True)

dataloader = DataLoader(dataset,batch_size=64, drop_last=True) # drop_last=True要设置为 true 不然batch不满会报错

class linear(torch.nn.Module):
    def __init__(self):
        super(linear, self).__init__()
        self.linear1 = Linear(196608, 10) # 第一项为输入的元素个数（这里为该图像所有的元素个数 即flatten后的结果）第二项为线性化后保留的元素个数

    def forward(self, input):
        output = self.linear1(input)
        return output


line = linear()

for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    output = torch.flatten(imgs) # 将tensor数据转为一维数据
    print(output.shape)
    output = line(output)
    print(output.shape)

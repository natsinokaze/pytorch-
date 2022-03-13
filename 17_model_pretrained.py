# 修改官方提供的模型
import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)  # 当为False时它只负责加载网络模型 是默认的参数
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 当为True时需要进行下载 是已经训练好的
# vgg16模型最终输出的是1000个类 但CIFAR10只有10个类 因此需要进行修改 有两种方式进行修改 一种是直接将原来的Linear(in_features=4096, out_features=1000,
# bias=True)修改为Linear(in_features=4096, out_features=10, bias=True) 第二种是添加一行Linear(in_features=1000, out_features=10,
# bias=True)
print(vgg16_false)
dataset = torchvision.datasets.CIFAR10(root="./dataset", transform= torchvision.transforms.ToTensor(), train=False, download=True)
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))  # 第二种方式 其中classifier可以不写 写了classifier就将nn.Linear(1000, 10)添加到nn.Linear(1000, 10)这层中
print("-----------------------------------------------------")
print(vgg16_true)
vgg16_false.classifier[6] = nn.Linear(4096, 10) # 第一种方式
print(vgg16_false)

import torchvision
from torch.utils.tensorboard import SummaryWriter

# train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)

writer=SummaryWriter("logs")

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


train_set = torchvision.datasets.CIFAR10(root="./dataset", transform= dataset_transform, train=True, download=True)# 直接将pil转成tensor
test_set = torchvision.datasets.CIFAR10(root="./dataset", transform= dataset_transform, train=False, download=True)



for i in range(10):
    img, target= test_set[i]# test_set包含了pil格式数据和标签 每一个标签（序号）对应一个类（比如汽车、猫、狗）
    da = test_set.data[i]
    print(da)
    # print(test_set.classes[target]) # test_set.classes[target]及表示标签对应的类
    writer.add_image("example", img, i)

writer.close()
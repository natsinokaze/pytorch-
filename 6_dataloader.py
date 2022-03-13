import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# shuffle表示是否打乱 num_workers是多进程 等于0时表示主进程 windows一般用主进程 否则容易报错 drop_last=false表示不舍去除不尽的图片


# img, target=test_data[111]
# print(img.shape)
# print(target)
writer = SummaryWriter("dataloader_logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1

writer.close()
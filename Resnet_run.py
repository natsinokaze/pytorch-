import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import ResNet18

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
from torchvision import datasets, transforms

data_path = "D:\\deep_learning\\datasets\\CIFAR10\\"
cifar10_train = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))
cifar10_train = DataLoader(cifar10_train, batch_size=32, shuffle=True)

cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))
cifar10_val = DataLoader(cifar10_val, batch_size=32, shuffle=True)

device = torch.device('cuda')
model = ResNet18.model.to(device)
model.load_state_dict(torch.load('save_model/model_4_0.pth'))
criteon = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)# first 1e-3
model.load_state_dict(torch.load('save_model/model_2_250.pth'))
for epoch in range(2,1000):

    model.train()
    for batchidx, (x, label) in enumerate(cifar10_train):
        x, label = x.to(device), label.to(device)

        logits = model(x)
        loss = criteon(logits, label)

        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batchidx % 50 == 0:
            torch.save(model.state_dict(), f'save_model\\model_{epoch}_{batchidx}.pth')

        #
        print('epoch:',epoch,'    index:',batchidx,'   loss:',loss.item())

        # test
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar10_val:
                x, label = x.to(device), label.to(device)

                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct / total_num
            print('acc is:      ',acc)

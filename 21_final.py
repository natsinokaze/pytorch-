import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "picture/img.png"
image = Image.open(image_path)  # 获得PIL类型的数据
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)


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
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("save/model_save_35.pth")
image = torch.reshape(image, (1, 3, 32, 32))
image = image.cuda() # 因为模型是由GPU训练的 不写会报错
model.eval()

with torch.no_grad():
    output = model(image)
output = model(image)

print(output.argmax(1))

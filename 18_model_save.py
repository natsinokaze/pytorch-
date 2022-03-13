import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1 模型结构+模型参数
# torch.save(vgg16,"vgg16_savemethod1.pth")
# model = torch.load("vgg16_savemethod1.pth")


# 保存方式2，模型参数（官方参数） 能减小容量
torch.save(vgg16.state_dict(), "vgg16_savemethod2.pth") # 以字典的方式保存
model = torch.load("vgg16_savemethod2.pth") # 直接print(model)得到的是字典类型数据 因此要创建网络模型再导入参数

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_savemethod2.pth"))

print(model)
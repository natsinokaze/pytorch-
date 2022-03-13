from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_pil = Image.open("hymenoptera_data/train/ants/5650366_e22b7e1065.jpg")

writer = SummaryWriter("logs")

# ToTensor
trans_totensor = transforms.ToTensor()  # 创建一个ToTesnsor类的实例
img_tensor = trans_totensor(img_pil)  # 调用实例的方法
writer.add_image("TOTENSOR", img_tensor)

# Normalize 归一化
print(img_tensor[0][0][0])

trans_norm = transforms.Normalize([3, 4, 1], [6, 1, 2])  # 通过构造函数创建Normalize的实例 第一项为设定三个通道的平均值 第二个为标准差
img_norm = trans_norm(
    img_tensor)  # 注意trans_norm不支持pil格式   # 公式 output[channel] = (input[channel] - mean[channel]) / std[channel]
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 3)
print("---------------")

# resize
print(img_pil.size)
trans_resize = transforms.Resize((100, 100))  # 两个参数表示改变后高宽
img_resize = trans_resize(img_pil)  # resize后得到的结果仍是pil类型
img_resize = trans_totensor(img_resize)  # 进行类型转换
writer.add_image("Resize", img_resize, 0)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(100)  # 只有一个参数则表示等比缩放
trans_compose = transforms.Compose(
    [trans_totensor, trans_resize_2])  # 通过构造函数实例化对象 compose及表示将两种对象组合 以前中括号内是按先后顺序处理的所以这里顺序不能交换 但现在可以
img_resize_2 = trans_compose(img_pil)  # 这里表示的就是等比缩放且转换类型后的结果
writer.add_image("Resize_2", img_resize_2, 1)

# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop((25, 20)) # 注意如果输入的值比原图大则会报错 输入一个参数则为正方形 输入两个则为指定高宽
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img_pil)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()

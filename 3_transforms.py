from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "hymenoptera_data/train/ants/5650366_e22b7e1065.jpg"
img_pil = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()# 创建一个transforms中的ToTensor这个类的实例对象 """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.
tensor_img = tensor_trans(img_pil) # 调用实例化对象的方法 将pil类型转换为tensor类型 因为writer.add_image()不支持pil类型的数据

writer.add_image("tensor_type_img", tensor_img)

writer.close()

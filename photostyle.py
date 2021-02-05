

print("this program is developed by Alan ^||^ ")
print("github: https://github.com/AlanYangYi/GayPhotoStyleRecongnizer")
print("*"*10)




import numpy as np
import matplotlib.pyplot as plt
print("model is building.... please wait....")
import torch
from torch import nn
print("model is building.... please wait....")
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
import cv2
print("model is building.... please wait....")







vgg19 = models.vgg19(pretrained=False)
vgg = vgg19.features

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv = vgg
        self.classifier = nn.Sequential(
            nn.Linear(25088, 2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x=x.view(-1,25088)
        x = self.classifier(x)
        return x

print("model is building.... please wait....")
mycnn=torch.load("model.pth")
trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class CAM(nn.Module):
    def __init__(self,mycnn1):
        super(CAM, self).__init__()
        self.features_conv = mycnn1.conv[0:36]
        self.max_pool = mycnn1.conv[36]
        self.classifier = mycnn1.classifier
        self.gradients = None
        # 获取地图的钩子函数

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)
        # 注册钩子
        h = x.register_hook(self.activations_hook)
        # 对卷积后的输出使用最大值池化
        x = self.max_pool(x)
        x = x.view(-1, 25088)
        x = self.classifier(x)
        return x


    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)


def heatmap(x,net):
    for p in net.parameters():
        p.requires_grad = True


    cam = CAM(net)
    cam.eval()

    im = Image.open(x)

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    input_im = trans(im).unsqueeze(0)
    im_pre = cam(input_im)
    im_pre.max().backward()
    print("Grad-CAM is building.....")
    gradients = cam.get_activations_gradient()
    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = cam.get_activations(input_im).detach()
    for i in range(len(mean_gradients)):
        activations[:, i, :, :] *= mean_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    print("Grad-CAM is building.....")
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    img = cv2.imread(x)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    Grad_cam_img = heatmap * 0.4 + img
    Grad_cam_img = Grad_cam_img / Grad_cam_img.max()
    b, g, r = cv2.split(Grad_cam_img)
    Grad_cam_img = cv2.merge([r, g, b])
    plt.figure()
    plt.imshow(Grad_cam_img)
    plt.show()
print("model is building.... please wait....")
print("model is built completely ^||^")
print("是否需要打印模型？ 输入yes/no")
p=input()
if p=="yes":
    print(mycnn)
print("======================================")
print("======================================")
print("请输入需要判别照片的绝对路径:")
impath = input()
im = Image.open(impath)
im = im.convert("RGB")
input_im = trans(im).unsqueeze(0)
mycnn.eval()
out = mycnn(input_im)
zhinan = (out.detach().numpy().flatten()[0] * 100).round()
jiyou = (out.detach().numpy().flatten()[1] * 100).round()
print(f"该照片为直男照片的概率:{zhinan}%   该照片为基友照片的概率: {jiyou}%")
print("是否要显示类热激活图：输入 1 显示，输入 0 不显示   ")
print("ps:类热激活图：https://arxiv.org/abs/1610.02391")
h=input()
if h=='1':
    heatmap(impath,mycnn)

input("输入任意键退出")






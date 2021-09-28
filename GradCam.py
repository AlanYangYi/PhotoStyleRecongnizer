import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
import cv2
from torch import nn

def heatmap(x):
    vgg19 = models.vgg19(pretrained=False)
    vgg = vgg19.features

    class cnn(nn.Module):
        def __init__(self):
            super(cnn, self).__init__()
            self.conv = vgg
            self.classifier = nn.Sequential(
                nn.Linear(25088, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            x = self.conv(x)
            x = x.view(-1, 25088)
            x = self.classifier(x)
            return x

    mycnn = torch.load("model.pth")
    for p in mycnn.parameters():
        p.requires_grad = True

    class CAM(nn.Module):
        def __init__(self):
            super(CAM, self).__init__()
            self.features_conv = mycnn.conv[0:36]
            self.max_pool = mycnn.conv[36]
            self.classifier = mycnn.classifier
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

            # 获取梯度的方法

        def get_activations_gradient(self):
            return self.gradients

            # 获取卷机层输出的方法

        def get_activations(self, x):
            return self.features_conv(x)

    cam = CAM()
    cam.eval()

    im = Image.open(x)
    imarray = np.asarray(im) / 255.0

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    input_im = trans(im).unsqueeze(0)
    im_pre = cam(input_im)
    #print(im_pre)
    im_pre.max().backward()
    gradients = cam.get_activations_gradient()
    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # 获取图像在相应卷积层输出的卷积特征
    activations = cam.get_activations(input_im).detach()
    # m每个通道乘以相应的梯度均值
    for i in range(len(mean_gradients)):
        activations[:, i, :, :] *= mean_gradients[i]
    # 计算所有通道的均值输出得到热力图
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    # 可视化热力图
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

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian

from models.squeezenet import SqueezeNet
from models.ResNet import resnet18, resnet50

import torchvision.models as torchmodels

import pdb

class AdaptiveThresh_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        #return torch.from_numpy(np.zeros(grad_output.shape)).float().cuda()
        return torch.FloatTensor(grad_output.shape).zero_().cuda()

class AdaptiveThresh(nn.Module):
    def __init__(self):
        super(AdaptiveThresh, self).__init__()

    def forward(self, input):
        return AdaptiveThresh_function.apply(input)

class Floor_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, step):
        tensor = tensor.float()
        step = step.float()
        x = tensor / step
        x = x.long()
        return x.float() * step

    @staticmethod
    def backward(ctx, grad_output):
        return torch.FloatTensor(grad_output.shape).zero_().cuda(), None

class Floor_step(nn.Module):
    def __init__(self, step):
        super(Floor_step, self).__init__()
        self.step = torch.tensor(step)

    def forward(self, input):
        return Floor_function.apply(input, self.step)


class edge(nn.Module):
    def __init__(self):
        super(edge, self).__init__()

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])
        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

    def forward(self, img):

        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2 + grad_x_g**2 + grad_y_g**2 + grad_x_b**2 + grad_y_b**2)
        
        grad_mag_max = grad_mag.max()
        grad_mag = grad_mag.div(grad_mag_max.expand_as(grad_mag))
        return grad_mag

class edge_squeezeNet(nn.Module):
    def __init__(self, version=1.1, num_classes=2):
        super(edge_squeezeNet, self).__init__()
        self.edge = edge()
        self.squeeze = SqueezeNet(version=version, num_classes=num_classes)

    def forward(self, x):
        x = self.edge(x)
        x = self.squeeze(x)
        return x

class edge_resnet18(nn.Module):
    def __init__(self, num_classes=2):
        super(edge_resnet18, self).__init__()
        self.edge = edge()
        for param in self.edge.parameters():
            param.requires_grad = False
        # self.Af = Floor_step(0.3)
        self.bn = nn.BatchNorm2d(1)
        self.resnet18 = resnet18(num_classes=num_classes)

    def forward(self, x):
        x = self.edge(x)
        # x = self.Af(x)
        x = self.bn(x)
        x = self.resnet18(x)
        return x

class edge_resnet50(nn.Module):
    def __init__(self, num_classes=2):
        super(edge_resnet50, self).__init__()
        self.edge = edge()
        for param in self.edge.parameters():
            param.requires_grad = False
        # self.Af = Floor_step(0.3)
        self.bn = nn.BatchNorm2d(1)
        self.resnet50 = resnet50(num_classes=num_classes)

    def forward(self, x):
        x = self.edge(x)
        # x = self.Af(x)
        x = self.bn(x)
        x = self.resnet50(x)
        return x

class Contrastive_res50_fe(nn.Module):
    def __init__(self, n_channels=3):
        super(Contrastive_res50_fe, self).__init__()
        self.resnet50 = resnet50(pretrained=True, n_channels=n_channels)
        self.res50_conv = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc = nn.Linear(2048, 2)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        x = self.res50_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)        
        output2 = self.forward_once(input2)
        return output1, output2

class Contrastive_res50(nn.Module):
    def __init__(self, con_res50_fe, num_classes=2):
        super(Contrastive_res50, self).__init__()
        self.constract_resnet50_fe = con_res50_fe
        self.fc = nn.Linear(con_res50_fe.num_feature, num_classes)

    def forward(self, x):
        x, _ = self.constract_resnet50_fe(x, x)
        x = self.fc(x)
        return x

def resnet50_ori_old():
  model = resnet50(n_channels=3, num_classes=2)
  model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model)

  return model

class resnet50_ori(nn.Module):
    def __init__(self, n_channels=3, num_classes=2, fe_branch=False, isPretrain=False):
        super(resnet50_ori, self).__init__()
        self.fe_branch = fe_branch
        _resnet50 = resnet50(pretrained=isPretrain, n_channels=n_channels)
        self.res50_conv = nn.Sequential(*list(_resnet50.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)
        #self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        
        x = self.res50_conv(x)
        res50_fe = x.view(x.size(0), -1)
        #res50_fe_bn = self.bn(res50_fe)
        x = self.fc(res50_fe)
        if self.fe_branch:
            return x, res50_fe
        else:
            return x
        

class resnet18_ori(nn.Module):
    def __init__(self, n_channels=3, num_classes=2, fe_branch=False, isPretrain=False):
        super(resnet18_ori, self).__init__()
        self.fe_branch = fe_branch
        _resnet18 = resnet18(pretrained=isPretrain, n_channels=n_channels)
        self.res18_conv = nn.Sequential(*list(_resnet18.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        #self.bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = self.res18_conv(x)
        res18_fe = x.view(x.size(0), -1)
        #res18_fe_bn = self.bn(res18_fe)
        x = self.fc(res18_fe)
        if self.fe_branch:
            return x, res18_fe
        else:
            return x
        

def test():
    import cv2
    img = cv2.imread('../test_img/bird.png')
    img = np.expand_dims(img, axis=0)
    img = np.transpose(img, (0, 3, 1, 2))
    img = img.astype('float')
    img = img / 255
    img_t = torch.from_numpy(img).float().cuda()

    Edge = edge()
    Edge.cuda()
    Af = Floor_step(0.2)
    Af.cuda()
    out_t = Af(Edge(img_t))
    img_out = out_t[0, 0].detach().cpu().numpy()
    img_out = (img_out) * 255
    img_out = img_out.astype('uint8')
    cv2.imwrite('./test_out.png', img_out)

def test_contrast():
    num_feature = 5
    con_res50_fe = Contrastive_res50_fe(num_feature=num_feature)
    model = Contrastive_res50(con_res50_fe)
    print(model)
    

if __name__ == '__main__':
    test_contrast()
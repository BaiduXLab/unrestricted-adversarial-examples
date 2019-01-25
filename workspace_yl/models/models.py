import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian

from models.squeezenet import SqueezeNet
from models.ResNet import resnet18

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

    
    

if __name__ == '__main__':
    test()
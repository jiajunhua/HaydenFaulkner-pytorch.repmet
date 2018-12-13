import torch.nn as nn
import torch.nn.functional as F


# Used for MNIST
class BasicCNN(nn.Module):
    def __init__(self, num_filters=[32, 64],
                 c_kernel_size=[5, 5], c_stride=[1, 1], c_padding=[2, 2], c_dilation=[1, 1],
                 p_kernel_size=[2, 2], p_stride=[2, 2], p_padding=[0, 0], p_dilation=[1, 1]):
        super(BasicCNN, self).__init__()
        self.blocks = []
        for i in range(len(num_filters)):
            if i == 0:
                in_num_filters = 1
            else:
                in_num_filters = num_filters[i-1]

            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_num_filters, num_filters[i], kernel_size=c_kernel_size[i], stride=c_stride[i], padding=c_padding[i], dilation=c_dilation[i]),
                # nn.BatchNorm2d(num_filters[i]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=p_kernel_size[i], stride=p_stride[i], padding=p_padding[i], dilation=p_dilation[i])))

    def forward(self, x, norm=False):
        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        for b in self.blocks:
            x = b(x)

        return x

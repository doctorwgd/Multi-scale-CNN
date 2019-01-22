import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Conv2d_4ch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d_4ch, self).__init__()
        padding1 = int((kernel_size + (kernel_size-1)*(1-1) ) / 2) if same_padding else 0
        padding2 = int((kernel_size + (kernel_size-1)*(2-1) ) / 2) if same_padding else 0
        padding3 = int((kernel_size + (kernel_size-1)*(3-1) ) / 2) if same_padding else 0
        padding4 = int((kernel_size + (kernel_size-1)*(4-1) ) / 2) if same_padding else 0
        
        #padding = dilation
        self.conv0 = nn.Conv2d(in_channels, 32, kernel_size, stride, padding = padding1,  dilation=1)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size, stride, padding = padding1,  dilation=1)
        self.conv2 = nn.Conv2d(in_channels, 64, kernel_size, stride, padding = padding2,  dilation=2)
        self.conv3 = nn.Conv2d(in_channels, 64, kernel_size, stride, padding = padding3,  dilation=3)
        self.conv4 = nn.Conv2d(in_channels, 64, kernel_size, stride, padding = padding4,  dilation=4)
        self.conv5 = nn.Conv2d(64, out_channels, kernel_size, stride, padding = padding1,  dilation=1)
        self.conv6 = nn.Conv2d(64, out_channels, kernel_size, stride, padding = padding2,  dilation=2)
        self.conv7 = nn.Conv2d(64, out_channels, kernel_size, stride, padding = padding3,  dilation=3)
        self.conv8 = nn.Conv2d(64, out_channels, kernel_size, stride, padding = padding4,  dilation=4)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.bn64 = nn.BatchNorm2d(64, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        #print(x.shape)
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x3 = self.relu(x3)
        x4 = self.relu(x4)
        x5 = self.conv5(x1)
        x6 = self.conv6(x2)
        x7 = self.conv7(x3)
        x8 = self.conv8(x4)
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        x = torch.cat((x0,x5,x6,x7,x8),1)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv2d_3ch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d_3ch, self).__init__()
        padding1 = int((kernel_size + (kernel_size-1)*(1-1) ) / 2) if same_padding else 0
        padding2 = int((kernel_size + (kernel_size-1)*(2-1) ) / 2) if same_padding else 0
        padding3 = int((kernel_size + (kernel_size-1)*(3-1) ) / 2) if same_padding else 0
        
        #padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding1,  dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding2,  dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding3,  dilation=3)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        #print(x.shape)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        x = torch.cat((x1,x2,x3),1)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Conv2d_2ch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d_2ch, self).__init__()
        padding1 = int((kernel_size + (kernel_size-1)*(1-1) ) / 2) if same_padding else 0
        padding2 = int((kernel_size + (kernel_size-1)*(2-1) ) / 2) if same_padding else 0
        #padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding1,  dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding2,  dilation=2)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        #print(x.shape)
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        x = torch.cat((x1,x2),1)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():        
        param = torch.from_numpy(np.asarray(h5f[k]))         
        v.copy_(param)


def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    if is_training:
        v = Variable(torch.from_numpy(x).type(dtype))
    else:
        v = Variable(torch.from_numpy(x).type(dtype), requires_grad = False, volatile = True)
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

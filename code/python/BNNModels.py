import torch
import torch.nn as nn
import torch.nn.functional as F
from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation
from Traintest_Utils import Scale
import binarizePM1

class Quantization1:
    def __init__(self, method):
        self.method = method
    def applyQuantization(self, input):
        return self.method(input)

binarizepm1 = Quantization1(binarizePM1.binarize)

class BNN_FASHION_CNN(nn.Module):
    def __init__(self):
        super(BNN_FASHION_CNN, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "BNN_FASHION_CNN"
        self.tlu_mode = None

        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=binarizepm1)

        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=binarizepm1)

        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=binarizepm1, bias=False)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=binarizepm1)

        self.fc2 = QuantizedLinear(2048, 10, quantization=binarizepm1, bias=False)
        # self.fc2 = nn.Linear(2048, 10)
        self.scale = Scale(init_value=1e-3)

    def forward(self, x):

        # conv2d block 1 does not use TLU (integer inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)
        x = F.max_pool2d(x, 2)

        # conv2d block 2
        if self.conv2.tlu_comp is not None:
            x = self.conv2(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.htanh(x)
            x = self.qact2(x)
        x = F.max_pool2d(x, 2)

        # fc block 1
        x = torch.flatten(x, 1)
        if self.fc1.tlu_comp is not None:
            x = self.fc1(x)
        else:
            x = self.fc1(x)
            x = self.bn3(x)
            x = self.htanh(x)
            x = self.qact3(x)

        # fc block 2 does not use TLU (no binarization)
        x = self.fc2(x)
        x = self.scale(x)
        return x

class BNN_CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(BNN_CIFAR10_CNN, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "BNN_CIFAR10_CNN"

        # block 1
        self.conv1 = QuantizedConv2d(3, 128, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, layerNr=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.qact1 = QuantizedActivation(quantization=binarizepm1)

        # block 2
        self.conv2 = QuantizedConv2d(128, 128, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, layerNr=2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.qact2 = QuantizedActivation(quantization=binarizepm1)

        # block 3
        self.conv3 = QuantizedConv2d(128, 256, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, layerNr=3, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.qact3 = QuantizedActivation(quantization=binarizepm1)

        # block 4
        self.conv4 = QuantizedConv2d(256, 256, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, layerNr=4, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.qact4 = QuantizedActivation(quantization=binarizepm1)

        # block 5
        self.conv5 = QuantizedConv2d(256, 512, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, layerNr=5, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.qact5 = QuantizedActivation(quantization=binarizepm1)

        # block 6
        self.conv6 = QuantizedConv2d(512, 512, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, layerNr=6, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.qact6 = QuantizedActivation(quantization=binarizepm1)

        # block 7
        self.fc1 = QuantizedLinear(8192, 1024, quantization=binarizepm1, layerNr=7, bias=False)
        self.bn7 = nn.BatchNorm1d(1024)
        self.qact7 = QuantizedActivation(quantization=binarizepm1)

        self.fc2 = QuantizedLinear(1024, 10, quantization=binarizepm1, layerNr=8, bias=False)
        self.scale = Scale(init_value=1e-3)

    def forward(self, x):

        # block 1 does not use TLU (integer inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)

        # block 2
        if self.conv2.tlu_comp is not None:
            x = self.conv2(x)
        else:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.htanh(x)
            x = self.qact2(x)
        x = F.max_pool2d(x, 2)

        # block 3
        if self.conv3.tlu_comp is not None:
            x = self.conv3(x)
        else:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.htanh(x)
            x = self.qact3(x)

        # block 4
        if self.conv4.tlu_comp is not None:
            x = self.conv4(x)
        else:
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.htanh(x)
            x = self.qact4(x)
        x = F.max_pool2d(x, 2)

        # block 5
        if self.conv5.tlu_comp is not None:
            x = self.conv5(x)
        else:
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.htanh(x)
            x = self.qact5(x)

        # block 6
        if self.conv6.tlu_comp is not None:
            x = self.conv6(x)
        else:
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.htanh(x)
            x = self.qact6(x)
        x = F.max_pool2d(x, 2)

        # block 7
        x = torch.flatten(x, 1)
        if self.fc1.tlu_comp is not None:
            x = self.fc1(x)
        else:
            x = self.fc1(x)
            x = self.bn7(x)
            x = self.htanh(x)
            x = self.qact3(x)

        x = self.fc2(x)
        x = self.scale(x)

        return x

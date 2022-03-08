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

q_train = True # quantization during training
q_eval = True # quantization during evaluation
binarizepm1 = Quantization1(binarizePM1.binarize)

class BNN_FMNIST(nn.Module):
    def __init__(self):
        super(BNN_FMNIST, self).__init__()
        self.htanh = nn.Hardtanh()
        self.name = "BNN_FMNIST"
        self.tlu_mode = None

        self.conv1 = QuantizedConv2d(1, 64, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, error_model=None, quantize_train=q_train, quantize_eval=q_eval, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.qact1 = QuantizedActivation(quantization=binarizepm1, quantize_train=q_train, quantize_eval=q_eval)

        self.conv2 = QuantizedConv2d(64, 64, kernel_size=3, padding=1, padding_mode = 'replicate', stride=1, quantization=binarizepm1, error_model=None, quantize_train=q_train, quantize_eval=q_eval, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.qact2 = QuantizedActivation(quantization=binarizepm1, quantize_train=q_train, quantize_eval=q_eval)

        self.fc1 = QuantizedLinear(7*7*64, 2048, quantization=binarizepm1, error_model=None, quantize_train=q_train, quantize_eval=q_eval, bias=False)
        self.bn3 = nn.BatchNorm1d(2048)
        self.qact3 = QuantizedActivation(quantization=binarizepm1, quantize_train=q_train, quantize_eval=q_eval)

        self.fc2 = QuantizedLinear(2048, 10, quantization=binarizepm1, error_model=None, quantize_train=q_train, quantize_eval=q_eval, bias=False)
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

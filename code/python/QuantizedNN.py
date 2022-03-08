import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np

import tluconv1d, tluconv2d

class Quantize(Function):
    @staticmethod
    def forward(ctx, input, quantization):

        output = input.clone().detach()
        output = quantization.applyQuantization(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

quantize = Quantize.apply


class ErrorModel(Function):
    @staticmethod
    def forward(ctx, input, error_model=None):
        output = input.clone().detach()
        output = error_model.applyErrorModel(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

apply_error_model = ErrorModel.apply

def check_quantization(quantize_train, quantize_eval, training):
    condition = ((quantize_train == True) and (training == True)) or ((quantize_eval == True) and (training == False)) or ((quantize_train == True) and (quantize_eval == True))

    if (condition == True):
        return True
    else:
        return False


class QuantizedActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedActivation"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.training = None
        super(QuantizedActivation, self).__init__(*args, **kwargs)

    def forward(self, input):
        output = None
        check_q = check_quantization(self.quantize_train,
         self.quantize_eval, self.training)
        if (check_q == True):
            output = quantize(input, self.quantization)
        else:
            output = input
        if self.error_model is not None:
            output = apply_error_model(output, self.error_model)
        return output


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedLinear"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.first_or_last_layer = kwargs.pop('first_or_last', None)
        self.training = None
        self.tlu_comp = None
        self.thresholds = None
        self.nr_xnor_gates = None
        self.nr_additional_samples = 0
        self.majv_shift = 0
        self.threshold_scaling = 0
        self.popc_acc = []
        self.popc_acc_activate = 0
        self.threshold_correction = 0
        self.popc_acc_normal = []
        self.popc_acc_normal_activate = None
        self.cycles = 0
        super(QuantizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)

            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight

            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, self.error_model)

            output = F.linear(input, quantized_weight)

            if self.popc_acc_normal_activate is not None:
                self.popc_acc_normal.append(output)

            # TLU-computation
            if self.tlu_comp is not None:
                # print("Executing with TLU: ", self.name)
                # print("Nr. of xnor gates: ", self.nr_xnor_gates)
                # print("Input shape: ", input.shape)
                # print("Weight shape: ", quantized_weight.shape)
                # print("Output shape: ", output.shape)
                # preparations:
                # wm_rows: 2048 (weight.shape[0])
                # wm_cols: 3136 (weight.shape[1])
                # im_cols = 1000 (input.shape[0])
                # output matrix: [im_cols x wm_rows]

                # nr. of neurons
                wm_row = quantized_weight.shape[0]
                # nr. of weights (and inputs)
                wm_col = quantized_weight.shape[1]
                # nr. of images in a batch
                im_col = input.shape[0]

                weight_b = quantized_weight
                # input_b = torch.transpose(input, 0, 1)
                input_b = input
                # print(self.thresholds)

                # prepare tensor for storing accumulated popcount results
                cycles = wm_col / self.nr_xnor_gates
                if wm_col % self.nr_xnor_gates != 0:
                    cycles += 1
                cycles = int(cycles)
                self.cycles = cycles
                popc_acc = torch.zeros(wm_row, cycles).cuda()

                output_b = torch.zeros_like(output)
                tluconv1d.customconv1d(input_b, weight_b, output_b, self.thresholds, popc_acc, self.nr_xnor_gates, self.nr_additional_samples, self.majv_shift, self.threshold_scaling, self.popc_acc_activate, self.threshold_correction)

                if self.popc_acc_activate == 1 and self.threshold_correction == 0:
                    self.popc_acc.append(popc_acc)

                # print("pop acc", popc_acc)
                # print("B:", output_b)
                # print("O: ", output)

                # correct = torch.eq(output_b, output)
                # correct = (~correct).sum().item()
                # print("correctness: ", correct)
                output = output_b
                # print("wm_row", wm_row)
                # print("wm_col", wm_col)
                # print("im_col", im_col)

                # cpu-based implementation is too slow
                # result = []
                # for i in range(wm_row):
                #     print("row: ", i)
                #     for j in range(im_col):
                #         cycle_counter = 0
                #         sub_popcnt = 0
                #         result_sub = []
                #         for k in range(wm_col):
                #             sub_popcnt += weight_b[i][k] * input_b[k][j]
                #             cycle_counter += 1
                #             if cycle_counter == n:
                #                 result_sub.append(sub_popcnt)
                #                 cycle_counter = 0
                #                 sub_popcnt = 0
                #         result.append(result_sub)
                # np_result = np.array(result)
            return output
        else:
            quantized_weight = None
            quantized_bias = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                quantized_bias = quantize(self.bias, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, self.error_model)
            return F.linear(input, quantized_weight, quantized_bias)


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedConv2d"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        self.error_model = kwargs.pop('error_model', None)
        self.quantize_train = kwargs.pop('quantize_train', True)
        self.quantize_eval = kwargs.pop('quantize_eval', True)
        self.first_or_last_layer = kwargs.pop('first_or_last', None)
        self.training = None
        self.tlu_comp = None
        self.thresholds = None
        self.nr_xnor_gates = None
        self.nr_additional_samples = 0
        self.majv_shift = 0
        self.threshold_scaling = 0
        self.popc_acc = []
        self.popc_acc_activate = 0
        self.threshold_correction = 0
        self.popc_acc_normal = []
        self.popc_acc_normal_activate = None
        self.cycles = 0
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, self.error_model)
            output = F.conv2d(input, quantized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # TLU-computation
            if self.tlu_comp is not None:
                # print("Executing with TLU: ", self.name)
                # print("Nr. of xnor gates: ", self.nr_xnor_gates)

                # get tensors in form of matrix multiplication
                h = input.shape[2]
                w = input.shape[3]
                kh = self.kernel_size[0]
                kw = self.kernel_size[1] # kernel size
                dh = self.stride[0]
                dw = self.stride[1] # stride
                size = int((h-kh+2*0)/dh+1)

                # unfold input
                input_b = F.unfold(input, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride).cuda()

                # unfold kernels
                weight_b = quantized_weight.view(self.out_channels,-1).cuda()

                # create output buffer
                output_b = torch.zeros(output.shape[0], weight_b.shape[0], input_b.shape[2]).cuda()

                # make the call to the cuda function
                tluconv2d.customconv2d(input_b, weight_b, output_b, self.thresholds, self.nr_xnor_gates, self.nr_additional_samples, self.majv_shift, self.threshold_scaling)

                # create the view that PyTorch expects
                output_b = output_b.view(output.shape[0], output.shape[1], output.shape[2], output.shape[3])

                output = output_b

            return output
        else:
            quantized_weight = None
            quantized_bias = None
            check_q = check_quantization(self.quantize_train,
             self.quantize_eval, self.training)
            # check quantization case
            if (check_q == True):
                quantized_weight = quantize(self.weight, self.quantization)
                quantized_bias = quantize(self.bias, self.quantization)
            else:
                quantized_weight = self.weight
                quantized_bias = self.bias
            # check whether error model needs to be applied
            if self.error_model is not None:
                quantized_weight = apply_error_model(quantized_weight, self.error_model)
                quantized_bias = apply_error_model(quantized_bias, self.error_model)
            # compute regular 2d conv
            output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np

import tluconv1d

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

            # TLU-computation
            if self.tlu_comp is not None:
                # print("Executing with TLU")
                print("Input shape: ", input.shape)
                # print("Weight shape: ", quantized_weight.shape)
                # print("Output shape: ", output.shape)
                # preparations:
                n = 64
                # wm_rows: 2048 (weight.shape[0])
                # wm_cols: 3136 (weight.shape[1])
                # im_cols = 1000 (input.shape[0])
                # output matrix: [im_cols x wm_rows]

                wm_row = quantized_weight.shape[0]
                wm_col = quantized_weight.shape[1]
                im_col = input.shape[0]

                weight_b = quantized_weight
                # input_b = torch.transpose(input, 0, 1)
                input_b = input

                output_b = torch.zeros_like(output)
                tluconv1d.customconv1d(input_b, weight_b, output_b)

                # print("B:", output_b)
                # print("O: ", output)

                correct = torch.eq(output_b, output)
                correct = (~correct).sum().item()
                print("correctness: ", correct)
                # print("wm_row", wm_row)
                # print("wm_col", wm_col)
                # print("im_col", im_col)

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
            output = F.conv2d(input, quantized_weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
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

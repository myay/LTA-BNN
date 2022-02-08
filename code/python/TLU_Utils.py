import torch
import torch.nn as nn
import argparse
import os
from datetime import datetime
import json

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

def extract_and_set_thresholds(model):
    # extract thresholds
    thresholds = []
    idx = 0
    for layer in model.children():
        idx += 1
        with torch.no_grad():
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # print("idx", idx)
                # print("Mean", layer.running_mean)
                # print("Var", layer.running_var)
                # print("gamma", layer.weight)
                # print("beta", layer.bias)
                threshold = layer.running_mean - torch.sqrt(layer.running_var+layer.eps)*layer.bias/layer.weight
                # print("thrshold", threshold[0])
                # print("thrshold", threshold[1])
                # print("thrshold", threshold[2])
                # print("thrshold", threshold[3])
                # threshold = torch.floor(threshold)
                thresholds.append(threshold)
                # print("threshold", threshold)
    # print("thresholds", thresholds)

    # first layer does not use TLU computations
    thresholds_nofirst = thresholds[1:]
    # print("thresholds_nofirst", thresholds_nofirst)
    # print("---")

    # set first and last layer flags
    model.conv1.first_or_last_layer = 1
    model.fc2.first_or_last_layer = 1

    # assign thresholds and computation types of layers
    idx = 0
    for layer in model.children():
        with torch.no_grad():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                if layer.first_or_last_layer is None:
                    layer.thresholds = thresholds_nofirst[idx]
                    idx += 1
                    layer.tlu_comp = 1

def print_layer_data(model):
    layer_idx = 0
    print("\n--- NN TLU CONFIGS ---\n")
    for layer in model.children():
        with torch.no_grad():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                print(">>> Layer: {}".format(layer_idx))
                print("Layer name: ", layer.name)
                print("TLU computation: ", layer.tlu_comp)
                print("#xnor gates: ", layer.nr_xnor_gates)
                layer_idx += 1

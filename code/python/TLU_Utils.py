import torch
import torch.nn as nn
import argparse
import os
from datetime import datetime
import json

from Traintest_Utils import test

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
                    # layer.tlu_comp = 1

def execute_with_TLU_FashionCNN(model, device, test_loader, xnor_gates_stat):
    # extract and set thresholds
    extract_and_set_thresholds(model)

    # activate TLU computation, set number of xnor gates, and nr of additional samples (0 by default) for each layer here
    model.conv2.tlu_comp = None # set to 1 to activate
    # model.conv2.nr_xnor_gates = 64
    model.conv2.nr_additional_samples = 0
    model.conv2.majv_shift = 0
    model.conv2.threshold_scaling = 0

    model.fc1.tlu_comp = 1 # set to 1 to activate
    # model.fc1.nr_xnor_gates = 64
    model.fc1.nr_additional_samples = 0
    model.fc1.majv_shift = 0
    model.fc1.threshold_scaling = 0
    # model.fc1.popc_acc_activate = activate

    # conv1
    # xnor_gates = [2**x for x in range(2, 9)]
    # xnor_gates = [32, 64]

    # xnor_gates = [4*x for x in range(1, 65)]
    # majv_shifts = [m for m in range(2, 7)]
    # additional_samples = [0, 1, 2]

    # xnor_gates = [4*x for x in range(1, 65)]
    xnor_gates = [xnor_gates_stat]
    majv_shifts = [0]
    additional_samples = [0]

    # print("\n")
    for majv_shift in majv_shifts:
        print("\n --- MAJV-SHIFT --- \n", majv_shift)
        for additional_sample in additional_samples:
            all_accuracies = []
            for nr_xnor in xnor_gates:
                # conv2d settings
                model.conv2.nr_xnor_gates = nr_xnor
                model.conv2.nr_additional_samples = additional_sample
                model.conv2.majv_shift = majv_shift
                # fc settings
                model.fc1.nr_xnor_gates = nr_xnor
                model.fc1.nr_additional_samples = additional_sample
                model.fc1.majv_shift = majv_shift
                # print_layer_data(model)
                accuracy = test(model, device, test_loader, pr=None)
                # print(accuracy)
                all_accuracies.append(accuracy)
            print("\n>> Add. samples: {}, Majv-shift: {}".format(additional_sample, majv_shift))
            print("Accuracies: \n", all_accuracies)
            print("FOR TIKZ")
            for idx, exp in enumerate(xnor_gates):
                print_str = "({}, {})".format(xnor_gates[idx], all_accuracies[idx])
                print(print_str)

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

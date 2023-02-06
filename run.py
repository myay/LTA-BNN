from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import json
import sys
import os
import scipy
from datetime import datetime
sys.path.append("code/python/")

from scipy.stats import norm
import matplotlib.pyplot as plt

from Utils import parse_args, dump_exp_data, create_exp_folder, store_exp_data, print_tikz_data, get_model_and_datasets

from Traintest_Utils import train, test, Criterion, binary_hingeloss, Clippy

from TLU_Utils import extract_and_set_thresholds, execute_with_TLU_FashionCNN, print_layer_data, execute_with_TLU

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from BNNModels import BNN_VGG3, BNN_VGG3_TLUTRAIN, BNN_VGG7, BNN_VGG7_TLUTRAIN

# from resnet_bnn import ResNet, BasicBlock

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Training Process')
    parse_args(parser)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    available_gpus = [i for i in range(torch.cuda.device_count())]
    print("Available GPUs: ", available_gpus)
    gpu_select = args.gpu_num
    # use selected GPU
    torch.cuda.set_device(gpu_select)
    # GPU that is currently used
    print("Currently used GPU: ", torch.cuda.current_device())

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    nn_model, dataset1, dataset2 = get_model_and_datasets(args)

    model = None
    # LTA processing cannot be applied to skip connections
    # if args.model == "ResNet18":
    #     # TODO: change model initialization
    #     model = nn_model(BasicBlock, [2, 2, 2, 2]).to(device)
    # else:
    model = nn_model().to(device)

    # set error model
    if args.error_prob is not None:
        model.error_model.p = args.error_prob

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    current_xc = args.nr_xnor_gates
    print("\n--- XNOR GATES: ", current_xc)
    if args.error_prob is not None:
        print("--- ERROR RATE: ", args.error_prob)

    # nr of xnor gates = 1 is reserved for "no lta computations"
    if current_xc != 1:
        model.lta_mode = args.lta_mode
        for layer in model.children():
            if isinstance(layer, (QuantizedConv2d, QuantizedLinear)):
                layer.nr_xnor_gates = current_xc

    # create experiment folder and file
    # to_dump_path = create_exp_folder(model)
    # if not os.path.exists(to_dump_path):
    #     open(to_dump_path, 'w').close()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Clippy(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # load training state or create new model
    if args.load_training_state is not None:
        print("Loaded training state: ", args.load_training_state)
        checkpoint = torch.load(args.load_training_state)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    time_elapsed = 0
    times = []
    # taining loop
    if args.train_model is not None:
        for epoch in range(1, args.epochs + 1):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            train(args, model, device, train_loader, optimizer, epoch)
            end.record()
            torch.cuda.synchronize()
            print("Run time (ms):", start.elapsed_time(end))
            times.append(start.elapsed_time(end))
            test(model, device, test_loader)
            scheduler.step()

    if args.test_error:
        all_accuracies = test_error(model, device, test_loader)
        to_dump_data = dump_exp_data(model, args, all_accuracies)
        store_exp_data(to_dump_path, to_dump_data)

    if args.save_model is not None:
        torch.save(model.state_dict(), "model_{}_{}.pt".format(args.save_model, current_xc))

    if args.save_training_state is not None:
        path = "model_checkpoint_{}.pt".format(args.save_training_state)
        torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print("Training state saved.")

    if args.load_model_path is not None:
        to_load = args.load_model_path
        print("Loaded model: ", to_load)
        model.load_state_dict(torch.load(to_load, map_location='cuda:0'))
        test(model, device, test_loader)

    if args.profile_time is not None:
        print_tikz_data(times)

if __name__ == '__main__':
    main()

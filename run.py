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

from BNNModels import BNN_VGG3, BNN_VGG3_TLUTRAIN, BNN_VGG7

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
    model = nn_model().to(device)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    current_xc = args.nr_xnor_gates
    print("\n--- XNOR GATES: ", current_xc)

    # nr of xnor gates = 1 is reserved for "no TLU computations"
    if current_xc != 1:
        model.tlu_mode = args.tlu_mode
        for layer in model.children():
            if isinstance(layer, (QuantizedConv2d, QuantizedLinear)):
                layer.nr_xnor_gates = current_xc

    # create experiment folder and file
    to_dump_path = create_exp_folder(model)
    if not os.path.exists(to_dump_path):
        open(to_dump_path, 'w').close()

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

    # xnor_gates_list = [64]#[4*x for x in range(1, 65)] #[4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256] #[4*x for x in range(1, 65)] #[2**x for x in range(2, 13)]
    # test(model, device, test_loader)
    # if args.tlu_mode is not None:
    #     # execute with TLU
    #     # execute_with_TLU_FashionCNN(model, device, test_loader, xnor_gates_list)
    #     # execute_with_TLU(model, device, test_loader, xnor_gates_list)
    #     execute_with_TLU(model, device, train_loader, xnor_gates_list)

    # print(model.eratel1)
    # erate1 = np.mean(np.array(model.eratel1))
    # erate2 = np.mean(np.array(model.eratel2))
    # print("--")
    # print(f"({xnor_gates_list[0]}, {erate1})")
    # print(f"({xnor_gates_list[0]}, {erate2})")
    # sets TLU-mode for each layer
    # extract_and_set_thresholds(model)
    # print_layer_data(model)
    # execute_with_TLU(model, device, test_loader, xnor_gates_list)
    # print_layer_data(model)
    # p2 = [2**x for x in range(2, 13)]
    # p2 = [3136]
    # threshold correction based on percentage
    '''
    ### threshold correction based on Fabio's method
    # get average popcounts of each neuron
    model.fc1.popc_acc_normal_activate = 1
    execute_with_TLU_FashionCNN(model, device, test_loader, xnor_gates_stat)
    list_nparray = []
    list_tensors = model.fc1.popc_acc_normal
    for tens in list_tensors:
        list_nparray.append(tens.cpu().numpy())
    np_list = np.array(list_nparray)
    average_over_batches = np_list.mean(axis=0)
    average_over_batches = average_over_batches.mean(axis=0)/model.fc1.cycles
    # print("avg_over_b", average_over_batches.shape)
    # get average popcount of each sliding window
    model.fc1.popc_acc_normal_activate = 0
    model.fc1.popc_acc_activate = 1
    execute_with_TLU_FashionCNN(model, device, test_loader, xnor_gates_stat)
    list_nparray = []
    list_tensors = model.fc1.popc_acc
    for tens in list_tensors:
        list_nparray.append(tens.cpu().numpy())
    np_list = np.array(list_nparray)
    np_list = (np_list.mean(axis=0)/1000) - 32
    np_list1 = average_over_batches.copy()
    # print("popc activations", np_list)
    for idx, threshold in enumerate(model.fc1.thresholds.cpu().numpy()):
        error = average_over_batches[idx] - np_list[idx]
        np_list[idx] = threshold + error
    # print("np_list", np_list.shape)
    print("with correction")
    model.fc1.thresholds_modified = torch.Tensor(np_list).cuda()
    model.fc1.threshold_correction = 1
    model.fc1.popc_acc_activate = 0
    execute_with_TLU_FashionCNN(model, device, test_loader, xnor_gates_stat)
    '''

    ### threshold correction based on Fabio's method (conv1)
    # model.conv2.popc_acc_normal_activate = 1
    # execute_with_TLU_FashionCNN(model, device, test_loader)
    # list_nparray = []
    # list_tensors = model.conv2.popc_acc_normal
    # for tens in list_tensors:
    #     list_nparray.append(tens.cpu().numpy())
    # np_list = np.array(list_nparray)
    # average_over_batches = np_list.mean(axis=0)
    # average_over_batches = average_over_batches.mean(axis=0)/model.conv2.cycles
    # # average_over_batches = average_over_batches.view([64, -1])
    # average_over_batches = torch.Tensor(average_over_batches).cuda().view(64,-1).cpu().numpy()
    # average_over_batches = average_over_batches.mean(axis=1)
    # # print("avg_over_b", average_over_batches)
    # # get average popcount of each sliding window
    # model.conv2.popc_acc_normal_activate = 0
    # model.conv2.popc_acc_activate = 1
    # execute_with_TLU_FashionCNN(model, device, test_loader)
    # list_nparray = []
    # list_tensors = model.conv2.popc_acc
    # for tens in list_tensors:
    #     list_nparray.append(tens.cpu().numpy())
    # np_list = np.array(list_nparray)
    # # np_list = (np_list.mean(axis=0)/(1000*196*64))
    # np_list = ((np_list.mean(axis=0)/(1000*196*64)) - 0.5)
    # # np_list1 = average_over_batches.copy()
    # # print("avg_over_b", average_over_batches)
    # # print("np_list", np_list)
    # # print("thresholds",model.conv2.thresholds.cpu().numpy())
    # for idx, threshold in enumerate(model.conv2.thresholds.cpu().numpy()):
    #     error = average_over_batches[idx] - np_list[idx]
    #     # print("error", error.shape)
    #     # np_list[idx] = threshold - error
    #     np_list[idx] = np.array([threshold for x in range(0,18)])
    # # print("np_list", np_list)
    # np_list /= model.conv2.cycles
    # print("with correction")
    # model.conv2.popc_acc = torch.Tensor(np_list).cuda()
    # model.conv2.threshold_correction = 1
    # model.conv2.popc_acc_activate = 0
    # execute_with_TLU_FashionCNN(model, device, test_loader)

    # TODO: Move into function
    # threshold correction based on my method
    '''
    model.fc1.threshold_correction = 0
    model.fc1.popc_acc_activate = 1
    execute_with_TLU_FashionCNN(model, device, train_loader, xnor_gates_stat)
    list_nparray = []
    list_tensors = model.fc1.popc_acc
    for tens in list_tensors:
        list_nparray.append(tens.cpu().numpy())
    np_list = np.array(list_nparray)
    # np_list = (np_list/(10000*xnor_gates_stat*2)) + 0.5
    np_list = (np_list.mean(axis=0)/(1000*xnor_gates_stat*2)) + 0.5
    np_list1 = np_list.copy()
    # print(np_list1.min())
    # print("thresholds: ", model.fc1.thresholds.cpu().numpy())
    for idx, threshold in enumerate(model.fc1.thresholds.cpu().numpy()):
        np_list[idx] *= threshold
        # for testing with original thresholds
        # np_list[idx] = np.array([threshold for x in range(0,model.fc1.cycles)])
    new_thresholds_tensor = torch.Tensor(np_list).cuda()/model.fc1.cycles
    # print("thresholds_modified", new_thresholds_tensor*model.fc1.cycles)
    # pass new thresholds to layer
    model.fc1.thresholds_modified = new_thresholds_tensor
    model.fc1.threshold_correction = 1
    model.fc1.popc_acc_activate = 0
    print("\n\nwith correction")
    print("cycles: ", model.fc1.cycles)
    execute_with_TLU_FashionCNN(model, device, test_loader, xnor_gates_stat)
    '''

    #conv2
    '''
    model.conv2.threshold_correction = 0
    model.conv2.popc_acc_activate = 1
    execute_with_TLU_FashionCNN(model, device, train_loader, xnor_gates_stat)
    list_nparray = []
    list_tensors = model.conv2.popc_acc
    for tens in list_tensors:
        list_nparray.append(tens.cpu().numpy())
    np_list = np.array(list_nparray)
    # np_list = (np_list/(10000*xnor_gates_stat*2)) + 0.5
    np_list = (np_list.mean(axis=0)/(1000*196*xnor_gates_stat*2)) + 0.5
    np_list1 = np_list.copy()
    # print(np_list1.min())
    # print("thresholds: ", model.fc1.thresholds.cpu().numpy())
    for idx, threshold in enumerate(model.conv2.thresholds.cpu().numpy()):
        np_list[idx] *= threshold
        # for testing with original thresholds
        #np_list[idx] = np.array([threshold for x in range(0,model.conv2.cycles)])
    new_thresholds_tensor = torch.Tensor(np_list).cuda()/model.conv2.cycles
    # print("thresholds_modified", new_thresholds_tensor*model.fc1.cycles)
    # pass new thresholds to layer
    model.conv2.thresholds_modified = new_thresholds_tensor
    model.conv2.threshold_correction = 1
    model.conv2.popc_acc_activate = 0
    print("\n\nwith correction")
    print("cycles: ", model.conv2.cycles)
    execute_with_TLU_FashionCNN(model, device, test_loader, xnor_gates_stat)
    '''


    # # extract values of sliding windows
    # xnor_gates_stat = [16]
    # model.fc1.threshold_correction = 0
    # model.fc1.popc_acc_activate = 1
    # execute_with_TLU_FashionCNN(model, device, train_loader, xnor_gates_stat)
    # list_nparray = []
    # list_tensors = model.fc1.popc_acc
    # print(len(list_tensors))
    # for tens in list_tensors:
    #     list_nparray.append(tens.cpu().numpy())
    # # np_list = np.array(list_nparray)/(1000*2*196)
    # np_list = np.array(list_nparray)/(1000*2)
    # print(np_list.shape)
    # np_list = np.round(np_list)
    # # print(np_list)
    # # np_list1 = np_list1.mean(axis=0)/(1000)
    # print(np_list.max())
    # print(np_list.min())
    #
    # #plot histogram of values
    # mu, std = norm.fit(np_list.flatten())
    # s = np.random.normal(mu, std, 65)
    # curve = plt.hist(np_list.flatten(), bins=12, color='g')
    # xmin, xmax = plt.xlim()
    # x = np.arange(0, 65, 1, dtype=int)
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2)
    # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    # plt.title(title)
    # plt.savefig("distr_popc_paper_conv1.pdf", format="pdf")
    #
    # print(curve)
    # # bins = curve.get_xdata()
    # # counts = curve.get_ydata()
    #
    # # bins = curve[1]
    # # counts = curve[0]
    # bins, counts = np.unique(np_list.flatten(), return_counts=True)
    #
    # # counts, bins, bars = plt.hist(np_list.flatten(), bins=16, color='g')
    # # print(bins)
    # # print(counts)
    # # TODO: print bins and values for tikz
    # for bin, count in zip(bins, counts):
    #     print("{} {}".format(bin, count))
    # # Nr. of XNOR gates:  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    #
    # # max_test_size = 64
    # # test_error_partial(model, device, test_loader, max_test_size)
    # #'''
if __name__ == '__main__':
    main()

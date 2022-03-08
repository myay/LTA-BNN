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
from datetime import datetime
sys.path.append("code/python/")

from scipy.stats import norm
import matplotlib.pyplot as plt

from Utils import set_layer_mode, parse_args, dump_exp_data, create_exp_folder, store_exp_data

from Traintest_Utils import train, test, Criterion, binary_hingeloss, Clippy

from TLU_Utils import extract_and_set_thresholds, execute_with_TLU_FashionCNN, print_layer_data

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from BNNModels import BNN_FMNIST

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Training Process')
    parse_args(parser)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.FashionMNIST('data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.FashionMNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = BNN_FMNIST().to(device)

    # create experiment folder and file
    to_dump_path = create_exp_folder(model)
    if not os.path.exists(to_dump_path):
        open(to_dump_path, 'w').close()

    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Clippy(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    time_elapsed = 0
    times = []
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        since = int(round(time.time()*1000))
        #
        train(args, model, device, train_loader, optimizer, epoch)
        #
        time_elapsed += int(round(time.time()*1000)) - since
        print('Epoch training time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
        # test(model, device, train_loader)
        since = int(round(time.time()*1000))
        #
        test(model, device, test_loader)
        #
        time_elapsed += int(round(time.time()*1000)) - since
        print('Test time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
        # test(model, device, train_loader)
        scheduler.step()

    if args.test_error:
        all_accuracies = test_error(model, device, test_loader)
        to_dump_data = dump_exp_data(model, args, all_accuracies)
        store_exp_data(to_dump_path, to_dump_data)

    if args.save_model:
        torch.save(model.state_dict(), "fmnist_cnn_cel_1x.pt")

    #'''
    # load model
    # to_load = "models/train_tlu/fashion/fmnist_cnn_xnor_mhl_32.pt"
    to_load = "mnist_cnn_mhl.pt"
    print("Loaded model: ", to_load)
    model.load_state_dict(torch.load(to_load, map_location='cuda:0'))

    # execute with TLU
    execute_with_TLU_FashionCNN(model, device, test_loader)
    # p2 = [2**x for x in range(2, 13)]
    # p2 = [3136]
    # threshold correction based on percentage

    '''
    model.fc1.threshold_correction = 0
    model.fc1.popc_acc_activate = 1
    execute_with_TLU_layerwise(model, device, test_loader, activate=1)
    list_nparray = []
    list_tensors = model.fc1.popc_acc
    for tens in list_tensors:
        list_nparray.append(tens.cpu().numpy())
    np_list = np.array(list_nparray)
    np_list = (np_list.sum(axis=0)/640000) + 0.5
    for idx, threshold in enumerate(model.fc1.thresholds.cpu().numpy()):
        np_list[idx] *= threshold
    new_thresholds_tensor = torch.Tensor(np_list).cuda()
    # pass new thresholds to layer
    model.fc1.popc_acc = new_thresholds_tensor
    model.fc1.threshold_correction = 1
    model.fc1.popc_acc_activate = 0
    print("\n\nwith correction")
    execute_with_TLU_layerwise(model, device, test_loader, activate=0)
    '''

    # plot histogram of values
    # mu, std = norm.fit(np_list.flatten())
    # s = np.random.normal(mu, std, 10)
    # plt.hist(np_list.flatten(), bins=50, density=True, alpha=0.6, color='g')
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 1000)
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2)
    # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    # plt.title(title)
    # plt.savefig("distr_popc.pdf", format="pdf")

    # Nr. of XNOR gates:  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # max_test_size = 64
    # test_error_partial(model, device, test_loader, max_test_size)
    #'''
if __name__ == '__main__':
    main()

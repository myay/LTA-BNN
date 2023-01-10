from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import json
import sys
import os
from datetime import datetime
sys.path.append("code/python/")

from Utils import Scale, Clippy, set_layer_mode, parse_args, dump_exp_data, create_exp_folder, store_exp_data, Criterion, binary_hingeloss

from TLU_Utils import extract_and_set_thresholds, print_layer_data

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

# from BNNModels import BNN_FMNIST

import binarizePM1
import binarizePM1FI

# bit error case
#python3 run_fashion_bin_fi.py --batch-size=256 --epochs=5 --lr=0.001 --step-size=25 --test-error

# Move error models to different file
class SymmetricBitErrorsBinarizedPM1:
    def __init__(self, method, p):
        self.p = p
        self.method = method
    def updateErrorModel(self, p_updated):
        self.p = p_updated
    def resetErrorModel(self):
        self.p = 0
    def applyErrorModel(self, input):
        return self.method(input, self.p, self.p)

class Quantization1:
    def __init__(self, method):
        self.method = method
    def applyQuantization(self, input):
        return self.method(input)

binarizepm1 = Quantization1(binarizePM1.binarize)
binarizepm1fi = SymmetricBitErrorsBinarizedPM1(binarizePM1FI.binarizeFI, 0)

cel_train = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_train")
cel_test = Criterion(method=nn.CrossEntropyLoss(reduction="none"), name="CEL_test")

q_train = True # quantization during training
q_eval = True # quantization during evaluation

nr_xnor_const = [4,8,12,16,24,32,48,64,96,128,192,256]
# nr_xnor_const = [4,8]
current_xc = 4

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
        # !!!
        self.conv2.nr_xnor_gates = current_xc
        self.conv2.nr_additional_samples = 0
        self.conv2.majv_shift = 0

        self.fc1.nr_xnor_gates = current_xc
        self.fc1.nr_additional_samples = 0
        self.fc1.majv_shift = 0

    def forward(self, x):
        # !!!
        extract_and_set_thresholds(self)

        # conv2d block 1 does not use TLU (integer inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.htanh(x)
        x = self.qact1(x)
        x = F.max_pool2d(x, 2)

        # conv2d block 2
        xt1 = x
        self.conv2.tlu_comp = None
        xt1 = x.clone().detach()
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.htanh(x)
        x = self.qact2(x)
        # TLU-based execution
        self.conv2.tlu_comp = 1 # for training with errors
        x.data.copy_(self.conv2(xt1).data)
        x = F.max_pool2d(x, 2)

        # fc block 1
        x = torch.flatten(x, 1)
        xt2 = x
        self.fc1.tlu_comp = None
        xt2 = x.clone().detach()
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.htanh(x)
        x = self.qact3(x)
        # TLU-based execution
        self.fc1.tlu_comp = 1
        x.data.copy_(self.fc1(xt2).data)

        # fc block 2 does not use TLU (no binarization)
        x = self.fc2(x)
        x = self.scale(x)

        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    set_layer_mode(model, "train") # propagate informaton about training to all layers

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        criterion = Criterion(binary_hingeloss, "MHL_train", param=128)
        # criterion = Criterion(nn.CrossEntropyLoss(reduction="none"), "CEL")
        loss = criterion.applyCriterion(output, target).mean()
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     if args.dry_run:
        #         break


def test(model, device, test_loader, pr=1):
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers

    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    if pr is not None:
        print('\nAccuracy: {:.2f}%\n'.format(
            100. * correct / len(test_loader.dataset)))

    accuracy = 100. * (correct / len(test_loader.dataset))

    return accuracy

def test_error(model, device, test_loader):
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers
    perrors = [i/100 for i in range(10)]

    all_accuracies = []
    for perror in perrors:
        # update perror in every layer
        for layer in model.children():
            if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
                if layer.error_model is not None:
                    layer.error_model.updateErrorModel(perror)

        print("Error rate: ", perror)
        accuracy = test(model, device, test_loader)
        all_accuracies.append(
            {
                "perror":perror,
                "accuracy": accuracy
            }
        )

    # reset error models
    for layer in model.children():
        if isinstance(layer, (QuantizedActivation, QuantizedLinear, QuantizedConv2d)):
            if layer.error_model is not None:
                layer.error_model.resetErrorModel()
    return all_accuracies

def execute_with_TLU(model, device, test_loader):
    # extract and set thresholds
    extract_and_set_thresholds(model)

    # activate TLU computation and set number of xnor gates
    # for each layer here
    # !!!
    model.conv2.tlu_comp = 1 # set to 1 to activate
    model.fc1.tlu_comp = 1 # set to 1 to activate

    # xnor_gates_list = [2**x for x in range(2, 9)]
    xnor_gates_list = [4,8,12,16,24,32,48,64,96,128,192,256]
    # xnor_gates_list = [4,8]
    # xnor_gates = [4*x for x in range(1, 65)]

    all_accuracies = []
    for nr_xnor in xnor_gates_list:
        model.conv2.nr_xnor_gates = nr_xnor
        model.fc1.nr_xnor_gates = nr_xnor
        # print_layer_data(model)
        accuracy = test(model, device, test_loader, pr=None)
        # print(accuracy)
        all_accuracies.append(accuracy)

    # for idx, acc in enumerate(all_accuracies):
    #     print("XNOR gates: ", xnor_gates_list[idx])
    #     print("Accuracy: ", all_accuracies[idx])

    print("XNOR gates: ", xnor_gates_list)
    print("Accuracy: ", all_accuracies)
    # !!!
    model.conv2.nr_xnor_gates = current_xc
    model.fc1.nr_xnor_gates = current_xc

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

    for xnor_gates_n in nr_xnor_const:
        current_xc = xnor_gates_n
        print("--- XNOR GATES: ", current_xc)

        model = BNN_FMNIST().to(device)
        # !!!
        model.conv2.nr_xnor_gates = current_xc
        model.fc1.nr_xnor_gates = current_xc
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
            # print('Epoch training time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
            # test(model, device, train_loader)
            since = int(round(time.time()*1000))
            #
            # test(model, device, test_loader)
            # execute_with_TLU(model, device, test_loader)
            #
            time_elapsed += int(round(time.time()*1000)) - since
            # print('Test time elapsed: {}ms'.format(int(round(time.time()*1000)) - since))
            # test(model, device, train_loader)
            scheduler.step()

        if args.test_error:
            all_accuracies = test_error(model, device, test_loader)
            to_dump_data = dump_exp_data(model, args, all_accuracies)
            store_exp_data(to_dump_path, to_dump_data)

        if args.save_model:
            torch.save(model.state_dict(), "fmnist_cnn_xnor_mhl_{}.pt".format(current_xc))

        # load model
        # to_load = "mnist_cnn.pt"
        # print("Loaded model: ", to_load)
        # model.load_state_dict(torch.load(to_load, map_location='cuda:0'))

        # execute with TLU
        # execute_with_TLU_layerwise(model, device, test_loader)
        # p2 = [2**x for x in range(2, 13)]
        # p2 = [3136]
        execute_with_TLU(model, device, test_loader)
        # Nr. of XNOR gates:  [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        # max_test_size = 64
        # test_error_partial(model, device, test_loader, max_test_size)
if __name__ == '__main__':
    main()

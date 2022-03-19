import torch
import torch.nn as nn
import argparse
import os
from datetime import datetime
import json

from QuantizedNN import QuantizedLinear, QuantizedConv2d, QuantizedActivation

from Utils import set_layer_mode

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    set_layer_mode(model, "train") # propagate informaton about training to all layers

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = Criterion(binary_hingeloss, "MHL_train", param=128)
        # criterion = Criterion(nn.CrossEntropyLoss(reduction="none"), "CEL")
        loss = criterion.applyCriterion(output, target).mean()
        loss.backward()
        optimizer.step()
        if (batch_idx % args.log_interval == 0) and (args.silent is None):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, pr=1):
    model.eval()
    set_layer_mode(model, "eval") # propagate informaton about eval to all layers

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    if pr is not None:
        print('\nAccuracy: {:.2f}%\n'.format(
            100. * correct / len(test_loader.dataset)))

    accuracy = 100. * (correct / len(test_loader.dataset))

    return accuracy

def binary_hingeloss(yhat, y, b=128):
    #print("yhat", yhat.mean(dim=1))
    #print("y", y)
    y_enc = 2 * torch.nn.functional.one_hot(y, yhat.shape[-1]) - 1.0
    #print("y_enc", y_enc)
    l = (b - y_enc * yhat).clamp(min=0)
    #print(l)
    return l.mean(dim=1) / b

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Clippy(torch.optim.Adam):
    def step(self, closure=None):
        loss = super(Clippy, self).step(closure=closure)
        for group in self.param_groups:
            for p in group['params']:
                p.data.clamp(-1,1)
        return loss

class Criterion:
    def __init__(self, method, name, param=None):
        self.method = method
        self.param = param
        self.name = name
    def applyCriterion(self, output, target):
        if self.param is not None:
            return self.method(output, target, self.param)
        else:
            return self.method(output, target)

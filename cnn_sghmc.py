import platform
print('python_version ==', platform.python_version())
import torch
print('torch.__version__ ==', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import argparse
import numpy as np
import math
from sghmc import *
from evaluation import *
from model_zoo import *


'''set up hyperparameters of the experiments'''
parser = argparse.ArgumentParser(description='sghmc on CNN tested on CIFAR10 appending noise')
parser.add_argument('--train-batch-size', type=int, default=64)
parser.add_argument('--test-batch-size', type=int, default=10000)
parser.add_argument('--num-burn-in', type=int, default=30000)
parser.add_argument('--num-epochs', type=int, default=1000)
parser.add_argument('--evaluation-interval', type=int, default=50)
parser.add_argument('--eta-theta', type=float, default=1.7e-8)
parser.add_argument('--c-theta', type=float, default=0.01)
parser.add_argument('--prior-precision', type=float, default=1e-3)
parser.add_argument('--permutation', type=float, default=0.2)
parser.add_argument('--enable-cuda', action='store_true')
parser.add_argument('--device-num', type=int, default=2)
args = parser.parse_args()
print (args)


if torch.cuda.is_available():
    torch.cuda.set_device(args.device_num)


'''load dataset'''
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./cifar10-dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.train_batch_size, shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./cifar10-dataset', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, drop_last=True)


if __name__ == '__main__':

    model = CNN()
    cuda_availability = args.enable_cuda and torch.cuda.is_available()
    N = len(train_loader.dataset)
    num_labels = model.outputdim
    sampler = SGHMC(model, N, args.eta_theta, args.c_theta)
    if cuda_availability:
        model.cuda()
    print(model)
    nIter = 0
    tStart = time.time()
    estimator = FullyBayesian((len(test_loader.dataset), num_labels),\
                               model,\
                               test_loader,\
                               cuda_availability)
    acc = 0
    sampler.resample_momenta()

    for epoch in range(1, 1 + args.num_epochs):
        print ("#######################################################################################")
        print ("This is the epoch: ", epoch)
        print ("#######################################################################################")
        for i, (x, y) in enumerate(train_loader):
            batch_size = x.data.size(0)
            if args.permutation > 0.0:
                y = y.clone()
                y.data[:int(args.permutation*batch_size)] = torch.LongTensor(np.random.choice(num_labels, int(args.permutation*batch_size)))
            if cuda_availability:
                x, y = x.cuda(), y.cuda()
            model.zero_grad()
            model.train()
            yhat = model(x)
            loss = F.cross_entropy(yhat, y)
            for param in model.parameters():
                loss += args.prior_precision * torch.sum(param**2)
            loss.backward()
            '''update position and momentum'''
            sampler.update()
            nIter += 1
            '''take the point and resample the particles'''
            if nIter%args.evaluation_interval == 0:
                print('loss:{:6.4f}; tElapsed:{:6.3f}'.format(loss.data.item(),\
                                                              time.time() - tStart))
                if nIter >= args.num_burn_in:
                    acc = estimator.evaluation()
                print ('This is the accuracy: %{:6.2f}'.format(acc))
                sampler.resample_momenta()
                tStart = time.time()

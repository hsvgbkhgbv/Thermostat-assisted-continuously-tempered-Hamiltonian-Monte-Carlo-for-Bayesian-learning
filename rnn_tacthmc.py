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
from tacthmc import *
from evaluation import *
import os
from model_zoo import *


'''set up hyperparameters of the experiments'''
parser = argparse.ArgumentParser(description='tacthmc on LSTM tested on Fashion MNIST appending noise')
parser.add_argument('--train-batch-size', type=int, default=64)
parser.add_argument('--test-batch-size', type=int, default=10000)
parser.add_argument('--num-burn-in', type=int, default=30000)
parser.add_argument('--num-epochs', type=int, default=2000)
parser.add_argument('--evaluation-interval', type=int, default=50)
parser.add_argument('--eta-theta', type=float, default=1.7e-8)
parser.add_argument('--eta-xi', type=float, default=1.7e-10)
parser.add_argument('--c-theta', type=float, default=0.1)
parser.add_argument('--c-xi', type=float, default=0.1)
parser.add_argument('--gamma-theta', type=float, default=1)
parser.add_argument('--gamma-xi', type=float, default=1)
parser.add_argument('--prior-precision', type=float, default=1e-3)
parser.add_argument('--permutation', type=float, default=0.3)
parser.add_argument('--enable-cuda', action='store_true')
parser.add_argument('--device-num', type=int, default=7)
parser.add_argument('--tempering-model-type', type=int, default=1)
parser.add_argument('--load-tempering-model', action='store_true')
parser.add_argument('--tempering-model-filename', type=int)
parser.add_argument('--save-tempering-model', action='store_true')
parser.add_argument('--tempering-model-path')
args = parser.parse_args()
print (args)


if torch.cuda.is_available():
    torch.cuda.set_device(args.device_num)


'''load dataset'''
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./fashion-dataset', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.train_batch_size, shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./fashion-dataset', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, drop_last=True)


if __name__ == '__main__':

    model = LSTM()
    cuda_availability = args.enable_cuda and torch.cuda.is_available()
    N = len(train_loader.dataset)
    num_labels = model.outputdim
    if args.tempering_model_type == 1:
        temper_model_name = 'Metadynamics'
    elif args.tempering_model_type == 2:
        temper_model_name = 'ABF'
    sampler = TACTHMC(model, N, args.eta_theta, args.eta_xi, args.c_theta, args.c_xi, args.gamma_theta, args.gamma_xi, cuda_availability, temper_model=temper_model_name)
    if cuda_availability:
        model.cuda()
    sampler.resample_momenta()
    print(model)
    nIter = 0
    tStart = time.time()
    estimator = FullyBayesian((len(test_loader.dataset), num_labels),\
                               model,\
                               test_loader,\
                               cuda_availability)
    acc = 0
    if args.save_tempering_model:
        path = args.tempering_model_path
        if path[-1] != '/':
            path = path+'/'
        try:
            os.mkdir(path)
        except:
            print ('There exists a folder or no tempering model path is set up!')
    if args.load_tempering_model:
        try:
            sampler.temper_model.loader(path+temper_model_name+'.pt', args.enable_cuda)
        except:
            print ('No saved model or no tempering model path is found!')

    for epoch in range(1, 1 + args.num_epochs):
        print ("#######################################################################################")
        print ("This is the epoch: ", epoch)
        print ("#######################################################################################")
        if epoch%(0.1*args.num_epochs) == 0:
            if temper_model_name == 'Metadynamics':
                sampler.temper_model.offset()
            if args.save_tempering_model:
                try:
                    sampler.temper_model.saver(path)
                except:
                    print ('No tempering model path is set up!')
        for i, (x, y) in enumerate(train_loader):
            batch_size = x.data.size(0)
            if args.permutation > 0.0:
                y = y.clone()
                y.data[:int(args.permutation*batch_size)] = torch.LongTensor(np.random.choice(num_labels, int(args.permutation*batch_size)))
            if cuda_availability:
                x, y = x.cuda(), y.cuda()
            model.zero_grad()
            yhat = model(x)
            loss = F.cross_entropy(yhat, y)
            for param in model.parameters():
                loss += args.prior_precision * torch.sum(param**2)
            loss.backward()
            '''update params and xi'''
            sampler.update(loss)
            nIter += 1
            if nIter%args.evaluation_interval == 0:
                print ('xi:{:+7.4f}; fU:{:+.3E}; r_xi:{:+.3E}; loss:{:6.4f}; thermostats_param:{:6.3f}; thermostats_xi:{:6.3f}; tElapsed:{:6.3f}'.format(sampler.model.xi.item(),\
                                                                                                                                                         sampler.fU.item(),\
                                                                                                                                                         sampler.model.r_xi.item(),\
                                                                                                                                                         loss.data.item(),\
                                                                                                                                                         sampler.get_z_u(),\
                                                                                                                                                         sampler.get_z_xi(),\
                                                                                                                                                         time.time() - tStart))
                if abs(sampler.model.xi.item()) <= 0.85*sampler.standard_interval and nIter >= args.num_burn_in:
                    acc = estimator.evaluation()
                print ('This is the accuracy: %{:6.2f}'.format(acc))
                sampler.resample_momenta()
                tStart = time.time()

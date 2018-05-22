# TACTHMC

This project implements the algorithm of TACTHMC and demonstrates the experiments compared with SGHMC, SGNHT, ADAM and SGD WITH MOMENTUM. All of algorithms are implemented in Python 3.6, with Pytorch 0.4.0 and Torchvision 0.2.1, so please install the relevant dependencies before running the codes or invoking the function.

The suggested solution is to install Anaconda Python 3.6 version: https://www.anaconda.com/download/.

Then install the latest version of Pytorch and Torchvision by the command shown as below
```bash
pip install torch torchvision
```

Now, all of dependencies are ready and you can start running the codes.

## EXPERIMENTS

We have done three experiments based on MLP, CNN and RNN. All of tasks are classifications.

The task of MLP is on EMNIST. The task of CNN is on CIFAR-10. The task of LSTM is on Fashion-MNIST.

### MLP

- architecture:
  | architecture |
  | ------ |
  | 784    |
  | linear |
  | ReLU   |
  | 100    |
  | linear |
  | ReLU   |
  | 47     |

- dataset: EMNIST-BALANCED

- train_data_size: 112800

- test_data_size: 18800

- categories: 47

- batch_size: 128


CNN:
architecture: 32x32->Conv2D(kernel=3x3x3x16, padding=1x1, stride= 1x1)
              ->ReLU
              ->MaxPooling2D(kernel=2x2, stride=2x2)
              ->Conv2D(kernel=3x3x16x16, padding=1x1, stride=1x1)
              ->ReLU
              ->MaxPooling2D(kernel=2x2, stride=2x2)
              ->Flatten
              ->linear->ReLU
              ->100
              ->linear->ReLU
              ->10
              
dataset: CIFAR-10

train_data_size: 60000

test_data_size:10000

categories: 10

batch_size: 64


RNN:
architecture: 28->LSTMCell(input=28, output=128) x 28 times
              ->the output of the last time step
              ->ReLU
              ->linear
              ->ReLU
              ->64
              ->linear
              ->ReLU
              ->10
              
dataset: Fashion-MNIST

train_data_size: 60000

test_data_size:10000

categories: 10

batch_size: 64

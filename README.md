# TACTHMC

This project implements the algorithm of TACTHMC and demonstrates the experiments compared with SGHMC, SGNHT, ADAM and SGD WITH MOMENTUM. All of algorithms are implemented in Python 3.6, with Pytorch 0.4.0 and Torchvision 0.2.1, so please install the relevant dependencies before running the codes or invoking the functions.

The suggested solution is to install Anaconda Python 3.6 version: https://www.anaconda.com/download/.

Then install the latest version of Pytorch and Torchvision by the command shown as below
```bash
pip install torch torchvision
```

Now, all of dependencies are ready and you can start running the codes.


## EXPERIMENTS

We have done three experiments based on MLP, CNN and RNN. All of tasks are classifications.

The task of MLP is on EMNIST. The task of CNN is on CIFAR-10. The task of LSTM is on Fashion-MNIST.

In experiments, we randomly assign 0%, 20% and 30% labels for each batch of training data respectively so as to constitute a noisy training environment.


### MLP

- architecture: 784->linear->ReLU->100->linear->ReLU->47
- dataset: EMNIST-BALANCED
- train_data_size: 112800
- test_data_size: 18800
- categories: 47
- batch_size: 128


### CNN
- architecture: 32x32->Conv2D(kernel=3x3x3x16, padding=1x1, stride= 1x1)
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
- dataset: CIFAR-10
- train_data_size: 60000
- test_data_size:10000
- categories: 10
- batch_size: 64


### RNN
- architecture: 28->LSTMCell(input=28, output=128) x 28 times
                ->the output of the last time step
                ->ReLU
                ->linear
                ->ReLU
                ->64
                ->linear
                ->ReLU
                ->10           
- dataset: Fashion-MNIST
- train_data_size: 60000
- test_data_size:10000
- categories: 10
- batch_size: 64


## Running the Preliminary Experiments

### Our Methods (TACTHMC)
```bash
python cnn_tacthmc_append_noise.py --random-selection-percentage 0.2
python mlp_tacthmc_append_noise.py --random-selection-percentage 0.2
python rnn_tacthmc_append_noise.py --random-selection-percentage 0.2
```

### Baseline
-SGD with Momentum
```bash
python cnn_sgd_append_noise.py --random-selection-percentage 0.2
python mlp_sgd_append_noise.py --random-selection-percentage 0.2
python rnn_sgd_append_noise.py --random-selection-percentage 0.2
```

-Adam
```bash
python cnn_adam_append_noise.py --random-selection-percentage 0.2
python mlp_adam_append_noise.py --random-selection-percentage 0.2
python rnn_adam_append_noise.py --random-selection-percentage 0.2
```

-SGHMC
```bash
python cnn_sghmc_append_noise.py --random-selection-percentage 0.2
python mlp_sghmc_append_noise.py --random-selection-percentage 0.2
python rnn_sghmc_append_noise.py --random-selection-percentage 0.2
```

-SGNHT
```bash
python cnn_sgnht_append_noise.py --random-selection-percentage 0.2
python mlp_sgnht_append_noise.py --random-selection-percentage 0.2
python rnn_sgnht_append_noise.py --random-selection-percentage 0.2
```



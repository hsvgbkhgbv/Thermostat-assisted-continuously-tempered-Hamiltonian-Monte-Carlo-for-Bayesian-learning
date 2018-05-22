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


## Run Preliminary Experiments

### Our Methods (TACTHMC)
```bash
python cnn_tacthmc_append_noise.py --random-selection-percentage 0.2
python mlp_tacthmc_append_noise.py --random-selection-percentage 0.2
python rnn_tacthmc_append_noise.py --random-selection-percentage 0.2
```

### Baseline

**SGD with Momentum
```bash
python cnn_sgd_append_noise.py --random-selection-percentage 0.2
python mlp_sgd_append_noise.py --random-selection-percentage 0.2
python rnn_sgd_append_noise.py --random-selection-percentage 0.2
```

**Adam
```bash
python cnn_adam_append_noise.py --random-selection-percentage 0.2
python mlp_adam_append_noise.py --random-selection-percentage 0.2
python rnn_adam_append_noise.py --random-selection-percentage 0.2
```

**SGHMC
```bash
python cnn_sghmc_append_noise.py --random-selection-percentage 0.2
python mlp_sghmc_append_noise.py --random-selection-percentage 0.2
python rnn_sghmc_append_noise.py --random-selection-percentage 0.2
```

**SGNHT
```bash
python cnn_sgnht_append_noise.py --random-selection-percentage 0.2
python mlp_sgnht_append_noise.py --random-selection-percentage 0.2
python rnn_sgnht_append_noise.py --random-selection-percentage 0.2
```

In these experiments, we implement SGNHT and SGHMC, as well as invoke SGD and Adam from Pytorch directly.
The reference for SGHMC is: https://arxiv.org/pdf/1402.4102.pdf
The reference for SGNHT is: http://people.ee.duke.edu/~lcarin/sgnht-4.pdf

### Some Advanced Settings
```bash
  -h, --help                                                 # show this help message and exit
  --train-batch-size TRAIN_BATCH_SIZE                        # set up the training batch size (int)
  --test-batch-size TEST_BATCH_SIZE                          # set up the test batch size (please set the size of the whole test data) (int)
  --num-burn-in NUM_BURN_IN                                  # set up the number of iterations of burn-in (int)
  --num-epochs NUM_EPOCHS                                    # set up the total number of epochs for training (int)
  --evaluation-interval EVALUATION_INTERVAL                  # set up the interval of evaluation (int)
  --eta-u ETA_U                                              # set up the learning rate of parameters, which should be divided by the size of the whole training dataset (float)
  --eta-xi ETA_XI                                            # set up the learning rate of the tempering variable which is similar to that of parameters (float)
  --c-u C_U                                                  # set up the noise level of parameters (float)
  --c-xi C_XI                                                # set up the noise level of the tempering variable (float)
  --gamma-xi GAMMA_XI                                        # set up the value of thermal initia (float)
  --prior-precision PRIOR_PRECISION                          # set up the penalizer of L2-norm (float)
  --random-selection-percentage RANDOM_SELECTION_PERCENTAGE  # set up the percentage of random assignment on labels (float)
  --enable-cuda                                              # use cuda if available (action=true)
  --device-num DEVICE_NUM                                    # select an appropriate GPU for usage (int)
  --experiments-num EXPERIMENTS_NUM                          # set up the label for the experiment (int)
  --tempering-model-type TEMPERING_MODEL_TYPE                # set up the model type for the tempering variable (1 for Metadynamics/2 for ABF) (int)
  --load-tempering-model                                     # set up whether necessarily load pre-trained tempering model (action=true)
  --tempering-model-filename TEMPERING_MODEL_FILENAME        # set up the tempering model filename (int)
  --saving-tempering-model                                   # set up whether it is necessary to save the tempering model   
```

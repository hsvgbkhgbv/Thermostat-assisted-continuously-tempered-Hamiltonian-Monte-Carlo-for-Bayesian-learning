# Thermostat-assisted continuously-tempered Hamiltonian Monte Carlo for Bayesian learning (TACTHMC)

This project implements the algorithm of TACTHMC and demonstrates the experiments compared with SGHMC, SGNHT, ADAM and SGD WITH MOMENTUM.

The paper of TACTHMC is shown as: http://papers.nips.cc/paper/8266-thermostat-assisted-continuously-tempered-hamiltonian-monte-carlo-for-bayesian-learning.

All of algorithms are implemented in Python 3.6, with Pytorch 0.4.0 and Torchvision 0.2.1, so please install the relevant dependencies before running the codes or invoking the functions.

The suggested solution is to install Anaconda Python 3.6 version: https://www.anaconda.com/download/.

Then install the latest version of Pytorch and Torchvision by the command shown as below
```bash
pip install torch torchvision
```

Now, all of dependencies are ready and you can start running the codes.


## EXPERIMENTS

We have done three experiments based on MLP, CNN and RNN. All of tasks are classifications.

The task of MLP is on EMNIST. The task of CNN is on CIFAR-10. The task of RNN is on Fashion-MNIST.

In experiments, we assign random labels to 0%, 20% and 30% of each batch of training data respectively so as to constitute a noisy training environment.

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
                ->linear
                ->ReLU
                ->100
                ->linear
                ->ReLU
                ->10        
- dataset: CIFAR-10
- train_data_size: 60000
- test_data_size: 10000
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
- test_data_size: 10000
- categories: 10
- batch_size: 64


### Evaluation Methods

For the conventional optimization algorithms such as Adam and SGD, we use the point estimate to evaluate the performance.

$$
\theta_{MAP} = arg \max_{\theta} \log P(D_{train}, \theta)
$$

$$
eval = P(D_{test}| \theta_{MAP})
$$

For the sampling algorithms such as SGHMC, SGNHT and TACTHMC, we use the fully bayesian to evaluate the performance.

$$
P(\theta | D_{train}) = \frac{P(D_{train}, \theta)}{P(D_{train})}
$$

$$
eval = \int_{\theta} P(D_{test}| \theta) P(\theta | D_{train}) \ d\theta
$$


### Experimental Results

#### MLP on EMNIST

% permuted labels | 0%         | 20%        | 30%
----------------- | ---------- | ---------- | ----------
Adam              | 83.39%     | 80.27%     | 80.63%
SGD               | 83.95%     | 82.64%     | 81.70%
SGHMC             | 84.53%     | 82.62%     | 81.56%
SGNHT             | 84.48%     | 82.63%     | 81.60%
TACTHMC           | **84.85%** | **82.95%** | **81.77%**

#### CNN on CIFAR-10

% permuted labels | 0%         | 20%        | 30%
----------------- | ---------- | ---------- | ----------
Adam              | 69.53%     | 72.39%     | 71.05%
SGD               | 64.25%     | 65.09%     | 67.70%
SGHMC             | 76.44%     | 73.87%     | 71.79%
SGNHT             | 76.60%     | 73.86%     | 71.37%
TACTHMC           | **78.93%** | **74.88%** | **73.22%**

#### RNN on Fashion-MNIST

% permuted labels | 0%         | 20%        | 30%
----------------- | ---------- | ---------- | ----------
Adam              | 88.84%     | 88.35%     | 88.25%
SGD               | 88.66%     | 88.91%     | 88.34%
SGHMC             | 90.25%     | 88.98%     | 88.49%
SGNHT             | 90.18%     | 89.10%     | 88.58%
TACTHMC           | **90.84%** | **89.61%** | **89.01%**


## Run Preliminary Experiments

### Our Method (TACTHMC)
```bash
python cnn_tacthmc.py --permutation 0.2 --c-theta 0.1
python mlp_tacthmc.py --permutation 0.2 --c-theta 0.05
python rnn_tacthmc.py --permutation 0.2 --c-theta 0.15
```

### Baseline

**SGD with Momentum**
```bash
python cnn_sgd.py --permutation 0.2
python mlp_sgd.py --permutation 0.2
python rnn_sgd.py --permutation 0.2
```

**Adam**
```bash
python cnn_adam.py --permutation 0.2
python mlp_adam.py --permutation 0.2
python rnn_adam.py --permutation 0.2
```

**SGHMC**
```bash
python cnn_sghmc.py --permutation 0.2 --c-theta 0.1
python mlp_sghmc.py --permutation 0.2 --c-theta 0.1
python rnn_sghmc.py --permutation 0.2 --c-theta 0.1
```

**SGNHT**
```bash
python cnn_sgnht.py --permutation 0.2 --c-theta 0.1
python mlp_sgnht.py --permutation 0.2 --c-theta 0.1
python rnn_sgnht.py --permutation 0.2 --c-theta 0.1
```

In these experiments, we implement SGNHT and SGHMC, as well as invoke SGD and Adam from Pytorch directly.

The reference for SGHMC is: https://arxiv.org/pdf/1402.4102.pdf

The reference for SGNHT is: http://people.ee.duke.edu/~lcarin/sgnht-4.pdf

The reference for SGD is: http://leon.bottou.org/publications/pdf/compstat-2010.pdf

The reference for Adam is: https://arxiv.org/pdf/1412.6980.pdf


### Some Advanced Settings
```bash
-h, --help                                                 # show this help message and exit
--train-batch-size TRAIN_BATCH_SIZE                        # set up the training batch size (int)
--test-batch-size TEST_BATCH_SIZE                          # set up the test batch size (please set the size of the whole test data) (int)
--num-burn-in NUM_BURN_IN                                  # set up the number of iterations of burn-in (int)
--num-epochs NUM_EPOCHS                                    # set up the total number of epochs for training (int)
--evaluation-interval EVALUATION_INTERVAL                  # set up the interval of evaluation (int)
--eta-theta ETA_THETA                                      # set up the learning rate of parameters, which should be divided by the size of the whole training dataset (float)
--eta-xi ETA_XI                                            # set up the learning rate of the tempering variable which is similar to that of parameters (float)
--c-theta C_THETA                                          # set up the noise level of parameters (float)
--c-xi C_XI                                                # set up the noise level of the tempering variable (float)
--gamma-theta GAMMA_THETA                                  # set up the value of the thermal initia of parameters (float)
--gamma-xi GAMMA_XI                                        # set up the value of the thermal initia of the tempering variable (float)
--prior-precision PRIOR_PRECISION                          # set up the penalty parameter of L2-Regularizer or the precision of a Gaussian prior from the view of bayesian stats (float)
--permutation PERMUTATION                                  # set up the percentage of random assignments on labels (float)
--enable-cuda                                              # use cuda if available (action=true)
--device-num DEVICE_NUM                                    # select an appropriate GPU for usage (int)
--tempering-model-type TEMPERING_MODEL_TYPE                # set up the model type for the tempering variable (1 for Metadynamics/2 for ABF) (int)
--load-tempering-model                                     # set up whether necessarily load pre-trained tempering model (action=true)
--save-tempering-model                                     # set up whether it is necessary to save the tempering model (bool)
--tempering-model-path TEMPERING_MODEL_PATH                # set up the path for saving or loading the tempering model (str)
```

Here the tempering model is to handle the unexpected noise for the tempering variable occuring during the dynamics.


## Invoke API of TACTHMC

### Formal Procedures

1. Initialize an instance of the object TACTHMC such as
```bash
sampler = TACTHMC(self, model, N, eta_theta0, eta_xi0, c_theta0, c_xi0, gamma_theta0, gamma_xi0, enable_cuda, standard_area=0.1, gaussian_decay=1e-3, version='accurate', temper_model='Metadynamics')
```

``` model ``` means the model instance constructed by Pytorch, which should be an instance inherited from ``` nn.Module ```

``` N ``` means the size of training dataset, which should be an int

``` eta_theta0 ``` means the learning rate of parameters divided by ``` N ```, which should be a float

``` eta_xi0 ``` means the learning rate of the tempering variable divided by ``` N ```, which should be a float

``` c_theta0 ``` means the noise level of parameters, which should be a float

``` c_xi0 ``` means the noise level of the tempering variable, which should be a float

``` gamma_theta0 ``` means the noise level of parameters, which should be a float

``` gamma_xi0 ``` means the thermal initia of the tempering variable, which should be a float

``` enable_cuda ``` means whether GPU is available, which should be a boolean

``` standard_interval ``` means the half range of the interval that the effective system temperature is at unity, which should be a float

``` gaussian_decay ``` means the decayed height of the stacked Gaussian (which is only feasible when ``` temper_model='Metadynamics' ```), which should be a float

``` version ``` means how to assign thermostats to parameters, which can be selected from either ``` 'accurate' ``` or ``` 'approximate' ```

``` temper_model ``` means which model is selected as the tempering variable model, which can be selectd between ``` 'Metadynamics' ``` and ``` 'ABF' ```

2. Initialize an estimator such as

```bash
estimator = FullyBayesian((len(test_loader.dataset), num_labels), model, test_loader, cuda_availability)
```

``` test_loader ``` means the data loader, which should be an instance of ``` torch.utils.data.DataLoader ```

``` num_labels ``` means the number of labels of dataset, which should be an int

``` cuda_availability ``` means the FLAG to identify whether to use GPU, which should be a boolean

3. Initialize the momenta of parameters such as

```bash
sampler.resample_momenta()
```

4. Evaluate with training data and get ``` loss ```

``` loss ``` can be the output from any loss function in Pytorch

5. Update the parameters and the tempering variable such as

```bash
sampler.update(loss)
```

6. Periodically resample the momenta of parameters and evaluate with test data such as

```bash
sampler.resample_momenta()
if abs(sampler.model.xi.item()) <= 0.85*sampler.standard_interval and nIter >= num_burn_in:
    acc = estimator.evaluation()
```

``` nIter ``` means the current iteration number, which should be an int

``` num_burn_in ``` means the iterations of waiting for convergence of the algorithm, which shoud be an int

7. Go back to step 4

### Some Useful Utilities

```bash
sampler.get_z_theta()                               # get the norm of thermostats of parameters
sampler.get_z_xi()                                  # get the norm of thermostats of the tempering variable
sampler.get_fU()                                    # get the current force of potential w.r.t the tempering variable
sampler.temper_model.loader(filename, enable_cuda)  # load the pre-trained tempering model
sampler.temper_model.saver(nIter, path)             # save the current tempering model
sampler.temper_model.estimate_force(xi)             # estimate the force caused by the unexpected noise of the current tempering variable xi
sampler.temper_model.update(xi)                     # update the tempering model for the current xi when the model is Metadynamics
sampler.temper_model.update(xi, fcurr)              # update the tempering model for the current xi by the current force fcurr when the model is ABF
sampler.temper_model.offset()                       # offset the stacked Gaussian when the model is Metadynamics
```

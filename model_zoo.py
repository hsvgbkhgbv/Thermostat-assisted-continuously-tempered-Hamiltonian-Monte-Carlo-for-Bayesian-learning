import torch.nn as nn
import torch.nn.functional as F
import torch
import math


DIM_INPUT_CNN = 32
CHANNEL_INPUT_CNN = 3
OUT_CHANNEL_CONV1_CNN = 16
KERNEL_SIZE_CONV1_CNN = 3
STRIDE_CONV1_CNN = 1
PADDING_CONV1_CNN = 1
OUT_CHANNEL_CONV2_CNN = 16
KERNEL_SIZE_CONV2_CNN = 3
STRIDE_CONV2_CNN = 1
PADDING_CONV2_CNN = 1
KERNEL_SIZE_MAXPOOLING2D_CNN = 2
STRIDE_MAXPOOLING2D_CNN = 2
DIM_LINEAR1_CNN = 100
DIM_OUTPUT_CNN = 10
DIM_INPUT_RNN = 28
DIM_HIDDEN_LSTM_RNN = 128
DIM_HIDDEN_LINEAR_RNN = 64
DIM_OUTPUT_RNN = 10
DIM_INPUT_MLP = 784
DIM_HIDDEN_MLP = 100
DIM_OUTPUT_MLP = 47


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=CHANNEL_INPUT_CNN, \
                               out_channels=OUT_CHANNEL_CONV1_CNN, \
                               kernel_size=KERNEL_SIZE_CONV1_CNN, \
                               stride=STRIDE_CONV1_CNN, \
                               padding=PADDING_CONV1_CNN)
        self.pool = nn.MaxPool2d(kernel_size=KERNEL_SIZE_MAXPOOLING2D_CNN, \
                                 stride=STRIDE_MAXPOOLING2D_CNN)
        self.conv2 = nn.Conv2d(in_channels=OUT_CHANNEL_CONV1_CNN, \
                               out_channels=OUT_CHANNEL_CONV2_CNN, \
                               kernel_size=KERNEL_SIZE_CONV2_CNN, \
                               stride=STRIDE_CONV2_CNN, \
                               padding=PADDING_CONV2_CNN)
        self.outputdim1 = int(math.floor((DIM_INPUT_CNN + 2 * PADDING_CONV1_CNN - KERNEL_SIZE_CONV1_CNN) / STRIDE_CONV1_CNN + 1) / 2)
        self.outputdim2 = int(math.floor((self.outputdim1 + 2 * PADDING_CONV2_CNN - KERNEL_SIZE_CONV2_CNN) / STRIDE_CONV2_CNN + 1) / 2)
        self.linear1 = nn.Linear(OUT_CHANNEL_CONV2_CNN * self.outputdim2 * self.outputdim2, DIM_LINEAR1_CNN)
        self.linear2 = nn.Linear(DIM_LINEAR1_CNN, DIM_OUTPUT_CNN)
        self.outputdim = DIM_OUTPUT_CNN

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, OUT_CHANNEL_CONV2_CNN * self.outputdim2 * self.outputdim2)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(DIM_INPUT_RNN, DIM_HIDDEN_LSTM_RNN, 1)
        self.linear1 = nn.Linear(DIM_HIDDEN_LSTM_RNN, DIM_HIDDEN_LINEAR_RNN)
        self.linear2 = nn.Linear(DIM_HIDDEN_LINEAR_RNN, DIM_OUTPUT_RNN)
        self.outputdim = DIM_OUTPUT_RNN

    def forward(self, x):
        x = torch.squeeze(x)
        x = torch.transpose(x, 0, 1)
        h0 = x.data.new(1, x.size(1), DIM_HIDDEN_LSTM_RNN).zero_()
        c0 = x.data.new(1, x.size(1), DIM_HIDDEN_LSTM_RNN).zero_()
        _, (x,_) = self.lstm(x, (h0, c0))
        x = torch.squeeze(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(DIM_INPUT_MLP, DIM_HIDDEN_MLP)
        self.linear2 = nn.Linear(DIM_HIDDEN_MLP, DIM_OUTPUT_MLP)
        self.outputdim = DIM_OUTPUT_MLP

    def forward(self, x):
        x = x.view(- 1, DIM_INPUT_MLP)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

import numpy as np
import math
from math import log, e
from scipy.stats import *
import copy
import PIL
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''Conv-1D'''
'''
class Conv_EEG(nn.Module):
    def __init__(self):
        super(Conv_EEG, self).__init__()
        self.conv1 = nn.Conv1d(1, 4, 3, stride=1)
        self.bn1   = nn.BatchNorm1d(4)
        self.lr1   = nn.LeakyReLU(0.3)
        self.mxp   = nn.MaxPool1d(4, 1)
        self.conv2 = nn.Conv1d(4, 8, 3, stride=1)
        self.bn2   = nn.BatchNorm1d(8)
        self.lr2   = nn.LeakyReLU(0.3)
        self.conv3 = nn.Conv1d(8, 16, 3, stride=1)
        self.bn3   = nn.BatchNorm1d(16)
        self.lr3   = nn.LeakyReLU(0.3)
        self.fn    = nn.Flatten()
        self.ms    = nn.Mish()
        self.dp    = nn.Dropout(0.2)

        self.regressor = nn.Sequential(
                            nn.Linear(31760,3060),
                            nn.Mish(),
                            nn.Dropout(0.2),
                            nn.Linear(3060,256),
                            nn.Mish(),
                            nn.Dropout(0.1),
                            nn.Linear(256,1)
                            # nn.Mish(),
                            # nn.Linear(32, 1), 
                            # nn.ReLU()
                            # nn.Tanh() 
                            # nn.Sigmoid()
                            )


    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.lr1(output)
        output = self.mxp(output)
        # output = self.dp(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.lr1(output)
        output = self.mxp(output)
        # output = self.dp(output)


        output = self.conv3(output)
        output = self.bn3(output)
        output = self.lr1   (output)
        output = self.mxp(output)
        output = self.dp(output)


        # output = self.mxp(output)
        # decoded_output = self.decoder(encoded_output)
        output = self.fn(output)
        # print(output.shape)
        output = self.regressor(output)


        return output
'''




'''Transformer'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=16, max_len=8000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]  # Change from x.size(0) to x.size(1)
        return x


class Conv_EEG(nn.Module):
    def __init__(self, d_model=16, nhead=1, num_layers=1, dim_feedforward=64, output_dim=1):
        super(Conv_EEG, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, 1, seq_len)
        x = x.squeeze(1) # (batch_size, seq_len)
        x = x.unsqueeze(-1) # (batch_size, seq_len, 1)
        x = self.embedding(x) # (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2) # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x) # (seq_len, batch_size, d_model)
        x = x.permute(1, 2, 0) # (batch_size, d_model, seq_len)
        x = self.global_avg_pool(x) # (batch_size, d_model, 1)
        x = x.squeeze(2) # (batch_size, d_model)
        x = self.fc(x) # (batch_size, output_dim)
        return x

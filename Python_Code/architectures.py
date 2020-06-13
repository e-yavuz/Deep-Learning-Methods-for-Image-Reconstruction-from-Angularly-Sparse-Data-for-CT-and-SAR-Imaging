# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 17:58:32 2019

@author: seant
"""
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as F
TAU = 0.001 #CHANGE FOR SOFT-THREHSOLDING / SHRINKAGE
torch.device("cuda")
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   
def shrinkage(V,theta):
    shrink = torch.sign(V)*torch.max(torch.abs(V) - theta, torch.zeros_like(V))
    return shrink

class BidirectionalRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(BidirectionalRNN,self).__init__()
        
        #Try to change to compare RNN and LSTM - 32 and 33
        #self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size,num_layers = 1,bidirectional = True, batch_first = True)
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size,num_layers = 1,bidirectional = True, batch_first = True)
        self.linear = nn.Linear(hidden_size*2, output_size, )
        self.act = nn.Tanh()

    def forward(self,x):
        pred, hidden = self.rnn(x,None)
        pred = pred.cuda()
        pred = (self.linear(pred)).view(pred.data.shape[0], pred.data.shape[1], 1)
        pred = shrinkage(pred,TAU)
        return pred

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=128,
            hidden_size=64,         # rnn hidden unit
            num_layers=25,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 128*128)
        self.act = nn.ReLU()

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:,-1,:])#self.act(self.out(r_out[:, -1, :]))
        return out

class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 64,kernel_size = 3,padding = 3//2)
        self.batch_norm1 = nn.BatchNorm2d(1)
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64,kernel_size = 3,padding = 3//2)
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        self.conv3=nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3,padding = 3//2)
        self.batch_norm3 = nn.BatchNorm2d(64)
        
        self.conv4=nn.Conv2d(in_channels = 64, out_channels = 1,kernel_size = 3,padding = 3//2)
        
        #self.batch_norm4 = nn.BatchNorm2d(1)
    
    def forward(self,x):
        down_1 = self.conv1(x)
        #down_1 = self.batch_norm1(x)
        down_1 = F.relu(down_1)
        
        down_2 = self.conv2(down_1)
        down_2 = self.batch_norm2(down_2)
        down_2 = F.relu(down_2)
        
        down_3 = self.conv3(down_2)
        down_3 = self.batch_norm3(down_3)
        down_3 = F.relu(down_3)
        
        down_4 = self.conv4(down_3)
        #down_4 = self.batch_norm4(down_4)
        #down_4 = F.relu(down_4)
        
        return down_4





class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.down_conv1 = double_conv(1, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.up_conv3 = double_conv(256 + 512, 256)
        self.up_conv2 = double_conv(128 + 256, 128)
        self.up_conv1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, 1, 1)
        
        
    def forward(self, x):
        conv1 = self.down_conv1(x)
        x = self.maxpool(conv1)

        conv2 = self.down_conv2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.down_conv3(x)
        x = self.maxpool(conv3)   
        
        x = self.down_conv4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.up_conv3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.up_conv2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.up_conv1(x)
        
        out = self.conv_last(x)
        
        return out
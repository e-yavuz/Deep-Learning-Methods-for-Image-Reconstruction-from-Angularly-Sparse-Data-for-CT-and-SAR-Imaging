# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 20:45:04 2020

@author: eyavu
"""



# load numpy array from npy file
from numpy import load
import matplotlib.pyplot as plt

allow_pickle=True
# load array
data = load('rnn_10MBS_l2l11e-5_tau1e-3_size128_lmbdaincrease_DispersedPaddingRadontoRadon_train.npy')
data2 = load('rnn_10MBS_l2l11e-5_tau1e-3_size128_lmbdaincrease_DispersedPaddingRadontoRadon_test.npy')
# print the array
#print(data[161:]/12)
#data[161:] = data[161:]/7
#print(data)
plt.figure()
plt.title("RNN")
plt.plot(data[51:],label = 'Training Loss')
plt.plot(data2[50:],label = 'Test Loss')
plt.legend()
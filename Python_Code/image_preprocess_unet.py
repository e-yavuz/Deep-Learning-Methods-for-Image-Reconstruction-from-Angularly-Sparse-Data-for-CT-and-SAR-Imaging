# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:00:01 2020

@author: seant
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import pickle
from sklearn.preprocessing import normalize, StandardScaler

i = 1
train = []
ground_truth = []
path = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Data_Generative_Code/'

num_train = 200 #Number of training
num_test = 50 #Number of testing images
im_size = 128 #Image size NxN
"""
=========================== TRAINING IMAGES ==================================
"""
while i < num_train+1:
    print(i)
    ground_truth_img = np.array(Image.open(path + 'Image_Data/Train/Iradon_180/' + 
                                           'Image_' + str(i) + '_orignial.png'))
    train_img = np.array(Image.open(path + 'Image_Data/Train/Iradon_60/' + 
                                    'Image_' + str(i) + '_test.png'))
    ground_truth.append(ground_truth_img)
    train.append(train_img)
    i+=1
    

#Dump files into pickle    
X_train = []
X_test = []
scaler = StandardScaler() #unit-mean, std variance

#Vectorize images, apply transform to columns
ground_truth = scaler.fit_transform((np.array(ground_truth)).reshape(num_train,
                                    im_size*im_size).T)
train = scaler.fit_transform((np.array(train)).reshape(num_train,
                             im_size*im_size).T)

#Convert back to row vectors
ground_truth = ground_truth.T.reshape(num_train,128,128)
train = train.T.reshape(num_train,128,128)

#Dump using pickle
for i in range(len(ground_truth)):
    X_train.append([ground_truth[i].astype(np.float32),
                    train[i].astype(np.float32)])
with open(path + 'PickleDir/' + 'train_unet_RNNData.pkl', 'wb') as f:
    pickle.dump(X_train,f)

"""
=========================== TESTING IMAGES ====================================
"""
ground_truth_test = []
test = []
j=1
while j < num_test+1:
    print(j)
    ground_truth_img = np.array(Image.open(path + 'Image_Data/Test/Iradon_180/' + 
                                           'Image_' + str(j) + '_orignial.png'))
    test_img = np.array(Image.open(path + 'Image_Data/Test/Iradon_60/' + 
                                    'Image_' + str(j) + '_test.png'))
    ground_truth_test.append(ground_truth_img)
    test.append(test_img)
    j+=1
    
    
print("Size of truth train array after for loop: " + str(np.array(ground_truth).size))
print("Size of train train array after for loop: " + str(np.array(test).size))

ground_truth_test = scaler.fit_transform((np.array(ground_truth_test)).reshape(num_test,im_size*im_size).T)
test = scaler.fit_transform((np.array(test)).reshape(num_test,im_size*im_size).T)

ground_truth_test = ground_truth_test.T.reshape(num_test,im_size,im_size)
test = test.T.reshape(num_test,im_size,im_size)

for i in range(len(ground_truth_test)):
    X_test.append([ground_truth_test[i].astype(np.float32), test[i].astype(np.float32)])
    
with open(path + 'PickleDir/' + 'test_unet_RNNData.pkl','wb') as f:
    pickle.dump(X_test,f)
    
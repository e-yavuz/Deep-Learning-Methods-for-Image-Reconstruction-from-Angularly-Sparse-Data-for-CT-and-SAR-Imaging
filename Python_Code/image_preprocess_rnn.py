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

i = 0
train = []
ground_truth = []
path = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Data_Generative_Code/'
path2 = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Python_Sean/codes/Data_Generative_Code'
num_train = 400 #Number of training
num_test = 100 #Number of testing images
radon_standard = 185
truth_im_size = 180 #Image size NxN
train_im_size = 180 #Image size NxN
"""
=========================== TRAINING IMAGES ==================================
"""
while i < num_train:
    print(i)
    
    ground_truth_img = np.array(Image.open(path + 'Image_Data/SAR/RadonPAD_Ground_Truth/' + 
                                           'ground_truth_' + str(i+1) + '.jpg'))
    train_img = np.array(Image.open(path + 'Image_Data/SAR/RadonPAD_60View_Images/' + 
                                           'ground_truth_' + str(i+1) + '.jpg'))
    ground_truth.append(ground_truth_img)
    train.append(train_img)
    """
    ground_truth_img = np.array(Image.open(path + 'Image_Data/Train/Origin_images_set-25_drg3/' + 
                                           'Image_' + str(i+1) + '_orignial.png'))
    train_img = np.array(Image.open(path + 'Image_Data/Train/Test_images_set-25_drg3_iradon/' + 
                                           'Image_' + str(i+1) + '_test.png'))
    ground_truth.append(ground_truth_img)
    train.append(train_img)
    """
    i+=1
#Dump files into pickle    
X_train = []
X_test = []
scaler = StandardScaler() #unit-mean, std variance

#Vectorize images, apply transform to columns
ground_truth = scaler.fit_transform((np.array(ground_truth)).reshape(num_train,
                                    radon_standard*truth_im_size).T)
train = scaler.fit_transform((np.array(train)).reshape(num_train,
                             radon_standard*train_im_size).T)

#Convert back to row vectors
ground_truth = ground_truth.T
train = train.T

#Dump using pickle
for i in range(len(ground_truth)):
    X_train.append([ground_truth[i].astype(np.float32),
                    train[i].astype(np.float32)])
with open(path + 'PickleDir/' + 'train_rnn_May31SarRadonWithDispersionPad.pkl', 'wb') as f:
    pickle.dump(X_train,f)
 
"""
=========================== TESTING IMAGES ====================================
"""
ground_truth_test = []
test = []
j=400
while j < num_test+num_train:
    print(j)

    ground_truth_img = np.array(Image.open(path + 'Image_Data/SAR/RadonPAD_Ground_Truth/' + 
                                           'ground_truth_' + str(j+1) + '.jpg'))
    test_img = np.array(Image.open(path + 'Image_Data/SAR/RadonPAD_60View_Images/' + 
                                           'ground_truth_' + str(j+1) + '.jpg'))
    ground_truth_test.append(ground_truth_img)
    test.append(test_img)
    """
    ground_truth_img = np.array(Image.open(path + 'Image_Data/Test/Origin_images_set-10_drg3/' + 
                                           'Image_' + str(j+1) + '_orignial.png'))
    test_img = np.array(Image.open(path + 'Image_Data/Test/Test_images_set-10_drg3_iradon/' + 
                                           'Image_' + str(j+1) + '_test.png'))
    ground_truth_test.append(ground_truth_img)
    test.append(test_img)
    """
    j+=1
    

ground_truth_test = scaler.fit_transform((np.array(ground_truth_test)).reshape(num_test,radon_standard*truth_im_size).T)
test = scaler.fit_transform((np.array(test)).reshape(num_test,radon_standard*train_im_size).T)

ground_truth_test = ground_truth_test.T
test = test.T

for i in range(len(ground_truth_test)):
    X_test.append([ground_truth_test[i].astype(np.float32), test[i].astype(np.float32)])
    
with open(path + 'PickleDir/' + 'test_rnn_May31SarRadonWithDispersionPad.pkl','wb') as f:
    pickle.dump(X_test,f)
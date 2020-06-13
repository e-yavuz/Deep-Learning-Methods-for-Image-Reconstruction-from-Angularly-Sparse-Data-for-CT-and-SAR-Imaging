# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:17:57 2020

@author: eyavu
"""


import torch.cuda
import torch.utils.data
import torch.nn as nn

import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.autograd import Variable
from architectures import UNet, BidirectionalRNN, ConvolutionalNeuralNetwork
from PIL import Image  
import PIL
 
path = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Python_Sean/codes/Data_Generative_Code/PreTrained_Networks/'
pathimg = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Python_Sean/codes/Data_Generative_Code/PreTrained_Networks/Paper_Img/'
ext = 'CNN'
save_image = True;
network_name = 'cnn_l2tv_lmbda1e-7_tau1e-3_RNNData.pt'
torch.device("cuda")
torch.cuda.synchronize()
# ==========================  RNN  ===============================

#HIDDEN_SIZE = 6 #Number of hidden layers (iterations)
#INPUT_DIM = 1 #Image is vectorized, input dimension is 1
#model = BidirectionalRNN(INPUT_DIM, HIDDEN_SIZE, 1)

# ==========================  CNN  ===============================

model = ConvolutionalNeuralNetwork()

# ==========================  CNN  ===============================

#model = UNet()

model.cuda()
criterion = nn.MSELoss() #nn.MSELoss() #Loss function is MSE
model.load_state_dict(torch.load(path + network_name))
model.eval()

pikl = 'ImageDomainEllipse_PaperImage.pkl'
with open(pathimg + pikl,'rb') as f:
    testset = pickle.load(f)
#print(np.array(testset).shape)
test_loader = torch.utils.data.DataLoader(testset,batch_size = 1,
                                          shuffle = False)
print("Loading complete...")
with torch.no_grad():
    for data, noisy_data in test_loader:
        """
        print("hello")
        inp = Variable(noisy_data.reshape(1,noisy_data.shape[1],1).cuda())
        out = Variable(data.reshape(1,data.shape[1],1).cuda())
        
        pred = model(inp)
        j = 0
        plt.figure()
        img_truth = out.cpu().detach().numpy()[j].reshape(185,180)
        img_noise = inp.cpu().detach().numpy()[j].reshape(185,180)
        img_recon = pred.cpu().detach().numpy()[j].reshape(185,180)
        """
        
        inp = Variable(noisy_data.unsqueeze(1).cuda())
            
        out = Variable(data.unsqueeze(1).cuda() )
            
        pred = model(inp).cuda()
        j = 0
        plt.figure()
        img_truth = out.cpu().detach().numpy()[j].reshape(128,128)
        img_noise = inp.cpu().detach().numpy()[j].reshape(128,128)
        img_recon = pred.cpu().detach().numpy()[j].reshape(128,128)
        
        print("Network operation complete...")
    
    #Image Variable Assignment
    
    
    if save_image:
        plt.imsave(pathimg + 'Ellipse_L2TVRnnData_' + ext + '.png',img_recon,cmap = 'gray')
    #if save_image:
        #img_save = Image.fromarray(img_recon)
        #img_save.save('SAR_Recon_' + ext + '.png')
        
        
    plt.imshow(img_truth,cmap = 'gray')
    plt.title("Original image")
    plt.figure()
    plt.imshow(img_noise,cmap = 'gray')
    plt.title("Noisy image")
    plt.figure()
    plt.imshow(img_recon,cmap = 'gray')
    plt.title("Reconstructed image")
    plt.figure()
    plt.imshow(np.abs(img_truth-img_recon),cmap = 'gray')
    plt.title("Difference image")
    
    n_mse = np.linalg.norm((img_truth-img_recon)**2,'fro')/np.linalg.norm(img_truth**2,'fro')
    print(n_mse)
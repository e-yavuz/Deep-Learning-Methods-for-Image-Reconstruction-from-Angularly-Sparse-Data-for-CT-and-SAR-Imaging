# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:02:13 2020

@author: seant
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 19:33:09 2019

@author: seant
"""
import torch.cuda
import torch.utils.data
import torch.nn as nn

import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.autograd import Variable
from architectures import BidirectionalRNN, ConvolutionalNeuralNetwork

"""
============================== CONSTANT VARIABLES =============================
"""
torch.device("cuda")
torch.cuda.synchronize()
EPOCHS = 500  #Run for 2500 epochs
MBS = 64#100 #MBS is 100
LEARNING_RATE = 1E-4 #Learning Rate
HIDDEN_SIZE = 8 #Number of hidden layers (iterations)
INPUT_DIM = 1 #Image is vectorized, input dimension is 1
PATH = 'C:/Users/seant/Documents/Emre_Project/Code/Data_Generative_Code/' #path
DS = 'train_cnn.pkl' #dataset

"""
================================== DATASET ====================================
"""
num_batches =500/100 #Number of training batches (computing loss)
num_batch_test = 300/100 #Number of testing batches (computing loss)

#Obtain training dataset, initiate train loader. Shuffle the data
with open(PATH + DS, 'rb') as f:
    trainset = pickle.load(f)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=MBS,
                                           shuffle=True)

#Obtain testing dataset, initiate test loader. Shuffle the data
DS = 'test.pkl'
with open(PATH + DS,'rb') as f:
    testset = pickle.load(f)
#print(np.array(testset).shape)
test_loader = torch.utils.data.DataLoader(testset,batch_size = MBS,
                                          shuffle = True)


"""
============================== INSTANTIATE MODEL =============================
"""
#Use BidirectionalRNN with 8 layers. Send to cuda()
cnn= ConvolutionalNeuralNetwork()
cnn.cuda()
criterion = nn.MSELoss() #Loss function is MSE
optimizer = torch.optim.Adam(cnn.parameters(),lr= LEARNING_RATE)

lmbda = 1E-7 #Hyperparameter for l1
train_loss_plot = []  #Keep track of training loss
test_loss_plot = [] #Keep track of testing loss

torch.autograd.set_detect_anomaly = False #Prevents memory leaks...

"""
==================================== TRAINING =================================
"""
for epoch in range(EPOCHS):
    torch.autograd.set_detect_anomaly = False #Prevents leaks
    #Reset loss
    train_loss = 0 
    test_loss = 0
    for i,(data, noisy_data) in enumerate(train_loader):#i,(data,noisy_data) in enumerate(train_loader):
        cnn.train()
        
        hidden = None
        #Obtain inputs (nosiy data), output (ground truth), prediction
        inp = Variable(noisy_data.unsqueeze(1).cuda())
            
        out = Variable(data.unsqueeze(1).cuda() )
        pred = cnn(inp).cuda()
        
        #Compute l1 loss...Pytorch does not seem to work correctly
        l1_loss = torch.sum(torch.abs(pred))
        #Obtain loss
        loss = criterion(pred, out)# + lmbda*l1_loss
        train_loss += loss.item()
        loss.backward() #Backpropagate
        optimizer.step() #Step optimizer
        optimizer.zero_grad() #Zero-out gradient
    #Test dataset...ignore for now. Or add into image pre-processing and run
    if False: #change to True if want to run
        with torch.no_grad(): #Do not train when running test set
            for data,noisy_data in test_loader:
                hidden = None
                inp = Variable(noisy_data.reshape(MBS,noisy_data.shape[1],1).cuda())
            
                out = Variable(data.reshape(MBS,data.shape[1],1).cuda() )
                
                pred = cnn(inp).cuda()
                l1_loss = torch.sum(torch.abs(pred))
                #predictions.append(pred.data.numpy())
                loss = criterion(pred, out) + lmbda*l1_loss
                test_loss += loss.item()
        
    test_loss = test_loss/num_batch_test
    train_loss = train_loss/(num_batches)
    train_loss_plot.append(train_loss) 
    test_loss_plot.append(test_loss)
    if epoch % 1 == 0:
        print("EPOCH: " + str(epoch) + " Trainling Loss: " + str(train_loss) + "Test Loss: " + str(test_loss))

#Sample images + Normalized MSE
        
j = np.random.randint(0,64)

noisy_im = inp.cpu().detach().numpy()[j].reshape(128,128)
ground_truth = out.cpu().detach().numpy()[j].reshape(128,128)
im_reconstructed = pred.cpu().detach().numpy()[j].reshape(128,128)
#Plot 
plt.figure()

plt.imshow(ground_truth,cmap = 'gray')
plt.title("Noisy image")
plt.figure()
plt.imshow(noisy_im,cmap = 'gray')
plt.figure()
plt.imshow(im_reconstructed,cmap = 'gray')

#Compute Normalized MSE
n_mse = np.linalg.norm((ground_truth-im_reconstructed)**2,'fro')/np.linalg.norm(ground_truth**2,'fro')
print(n_mse)
#Pass through entire test set
if False:
    MBS = 300
    test_loader = torch.utils.data.DataLoader(testset,batch_size = MBS,shuffle = True)
    with torch.no_grad():
        for data, noisy_data in test_loader:
            inp = Variable(noisy_data.reshape(MBS,noisy_data.shape[1],1).cuda())
            out = Variable(data.reshape(MBS,data.shape[1],1).cuda())
            
            pred = cnn(inp)
    
    j = np.random.randint(0,100)
    plt.figure()
    
    plt.imshow(inp.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
    plt.title("Noisy image")
    plt.figure()
    plt.imshow(out.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
    plt.figure()
    plt.imshow(pred.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
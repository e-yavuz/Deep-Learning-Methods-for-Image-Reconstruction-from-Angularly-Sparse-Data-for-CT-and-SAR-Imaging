import torch.cuda
import torch.utils.data
import torch.nn as nn

import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.autograd import Variable
from architectures import BidirectionalRNN
print("Finished importing")

"""
============================== CONSTANT VARIABLES =============================
"""
torch.device("cuda")
torch.cuda.get_device_name()
torch.cuda.synchronize()
EPOCHS = 250  #Run for 2500 epochs
MBS = 25#100 #MBS is 100
LEARNING_RATE = 1E-3 #Learning Rate
Learning_Flag_1 = False
Learning_Flag_2 = False
Learning_Flag_3 = False
HIDDEN_SIZE = 8 #Number of hidden layers (iterations)
INPUT_DIM = 1 #Image is vectorized, input dimension is 1
PATH = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Python_Sean/codes/Data_Generative_Code/' #path


"""
================================== DATASET ====================================
"""
num_images = 400
num_tests = 100
num_batches = num_images/MBS #Number of training batches (computing loss)
num_batch_test = num_tests/MBS #Number of testing batches (computing loss)
#image_size = 128

model_pretrained_name = "rnn_25MBS_l2_tau1e-3_size128_SAR_DispersedPaddingRadontoRadon.pt"

PicklePath = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Data_Generative_Code/PickleDir/'
DS = 'train_rnn_May31SarRadonWithDispersionPad.pkl' #dataset
#Obtain training dataset, initiate train loader. Shuffle the data
with open(PicklePath + DS, 'rb') as f:
    trainset = pickle.load(f)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=MBS,
                                           shuffle=True)

#Obtain testing dataset, initiate test loader. Shuffle the data
DS = 'test_rnn_May31SarRadonWithDispersionPad.pkl'
with open(PicklePath + DS,'rb') as f:
    testset = pickle.load(f)
#print(np.array(testset).shape)
test_loader = torch.utils.data.DataLoader(testset,batch_size = MBS,
                                          shuffle = True)

"""
DS = 'train_rnn.pkl' #dataset
#Obtain training dataset, initiate train loader. Shuffle the data
with open(PATH + DS, 'rb') as f:
    trainset = pickle.load(f)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=MBS,
                                           shuffle=True)

#Obtain testing dataset, initiate test loader. Shuffle the data
DS = 'test_rnn.pkl'
with open(PATH + DS,'rb') as f:
    testset = pickle.load(f)
#print(np.array(testset).shape)
test_loader = torch.utils.data.DataLoader(testset,batch_size = MBS,
                                          shuffle = True)
"""

"""
============================== INSTANTIATE MODEL =============================
"""
#Use BidirectionalRNN with 8 layers. Send to cuda()
r = BidirectionalRNN(INPUT_DIM, HIDDEN_SIZE, 1)
r.cuda()
criterion = nn.MSELoss() #nn.MSELoss() #Loss function is MSE
optimizer = torch.optim.Adam(r.parameters(),lr= LEARNING_RATE)

lmbda = 1E-7 #Hyperparameter for l1 or TV
train_loss_plot = []  #Keep track of training loss
test_loss_plot = [] #Keep track of testing loss
"""
==================================== TRAINING =================================
"""
total_l2 = 50

def compute_total_variation_loss(y):      
        tv_h = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
        tv_w = torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])) 
        
        return (tv_h + tv_w)

torch.autograd.set_detect_anomaly(True)
train_loss = 1 
test_loss = 1
print("Starting Training...")
for epoch in range(EPOCHS):
        #Updating Learning Rate based upon loss
        if (total_l2 < 0.01 and Learning_Flag_3 != True):
            LEARNING_RATE = LEARNING_RATE/10
            lmbda =lmbda*10
            Learning_Flag_3 = True
            print("LR switched to " + str(LEARNING_RATE))
        elif(total_l2 < 0.05 and Learning_Flag_2 != True):
            LEARNING_RATE = LEARNING_RATE/10
            lmbda =lmbda*10
            Learning_Flag_2 = True
            print("LR switched to " + str(LEARNING_RATE))
        elif(total_l2 < 0.1 and Learning_Flag_1 != True):
            LEARNING_RATE = LEARNING_RATE/10
            Learning_Flag_1 = True
            print("LR switched to " + str(LEARNING_RATE))
        #Reset loss
        train_loss = 0 
        test_loss = 0
        total_l2 = 0
        total_reg = 0
        for data, noisy_data in train_loader:#i,(data,noisy_data) in enumerate(train_loader):
            r.train()
            
            hidden = None
            #Obtain inputs (nosiy data), output (ground truth), prediction
            inp = Variable(noisy_data.reshape(MBS,noisy_data.shape[1],1).cuda())
                
            out = Variable(data.reshape(MBS,data.shape[1],1).cuda() )
            pred = r(inp).cuda()
            
            #Compute l1 loss...Pytorch does not seem to work correctly
            #l1_loss = torch.abs( torch.sum(torch.abs(out)) -torch.sum(torch.abs(pred)) )
            #print(lmbda*l1_loss)
            #Calculate tv and l1 in refrence to original image - predicted image or something like that...
            #tv_loss = torch.abs(compute_total_variation_loss(out.view(MBS,185,180).unsqueeze(1)) - compute_total_variation_loss(pred.view(MBS,185,180).unsqueeze(1)))
            #print("tv_loss = " + str(lmbda*tv_loss.item()))
            l2_loss = criterion(pred,out) 
            #Obtain loss
            #print(l1_loss.item())
            loss = l2_loss# +(lmbda*tv_loss)
            total_l2 += l2_loss.item()
            #total_reg += lmbda*tv_loss.item()
            train_loss += l2_loss.item()# + lmbda*tv_loss.item()
            loss.backward() #Backpropagate
            optimizer.step() #Step optimizer
            optimizer.zero_grad() #Zero-out gradient
        #Test dataset...ignore for now. Or add into image pre-processing and run
        if True: #change to True if want to run
            
            
            with torch.no_grad(): #Do not train when running test set
                for data,noisy_data in test_loader:
                    hidden = None
                    inp = Variable(noisy_data.reshape(MBS,noisy_data.shape[1],1).cuda())
                
                    out = Variable(data.reshape(MBS,data.shape[1],1).cuda() )
                    
                    pred = r(inp).cuda()
                    #l1_loss = torch.sum(torch.abs(pred))
                    #predictions.append(pred.data.numpy())
                    loss = criterion(pred, out)# + lmbda*l1_loss
                    test_loss += loss.item()
                    
                    if(epoch%50 == 0):
                        j = np.random.randint(0,MBS)
                        path1 = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Data_Generative_Code/Image_Data/'

                        plt.imsave(path1 + 'Test/' + 'RNN-1/Noisy/' + 'Image_' + str(epoch) + '_noisy_25MBS_l2_tau1e-3_size128_SAR_DispersedPaddingRadontoRadon.png',
                                   inp.cpu().detach().numpy()[j].reshape(185,180),cmap = 'gray')
                        plt.imsave(path1 + 'Test/' + 'RNN-1/Ground_Truth/' + 'Image_' + str(epoch) + '_orignial_25MBS_l2_tau1e-3_size128_SAR_DispersedPaddingRadontoRadon.png',
                                   out.cpu().detach().numpy()[j].reshape(185,180),cmap = 'gray')
                        plt.imsave(path1 + 'Test/' + 'RNN-1/Reconstructed/' + 'Image_' + str(epoch) + '_reconstructed_25MBS_l2_tau1e-3_size128_SAR_DispersedPaddingRadontoRadon.png',
                                   pred.cpu().detach().numpy()[j].reshape(185,180),cmap = 'gray')
        
        test_loss = test_loss/num_batch_test
        train_loss = train_loss/(num_batches)
        total_l2 = total_l2/(num_batches)
        #total_reg = total_reg/(num_batches)
        train_loss_plot.append(train_loss) 
        test_loss_plot.append(test_loss)
        print("l2:" + str(total_l2))
        #print("regul:" + str(total_reg))
        if epoch % 1 == 0:
            print("EPOCH: " + str(epoch) + " Training Loss: " + str(train_loss) + " Testing Loss: " + str(test_loss))

print("Creating model loss plot...")
"""
x_dim_train_plot = []
i= 0
while i < EPOCHS:
    i +=1
    x_dim_train_plot.append(i)
"""
plt.figure()
plt.plot(train_loss_plot[2:],label = 'Training Loss')
plt.plot(test_loss_plot,label = 'Testing loss')
plt.legend()
train_loss_plot_numpy = np.asarray(train_loss_plot)
test_loss_plot_numpy = np.asarray(test_loss_plot)
np.save(('C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Loss_Data/' + str(model_pretrained_name[:-3]) + '_train.npy'), train_loss_plot_numpy)
np.save(('C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Loss_Data/' + str(model_pretrained_name[:-3]) + '_test.npy'), test_loss_plot_numpy)


#plt.plot(np.asarray(x_dim_train_plot), np.asarray(train_loss_plot))
print("Finished Training...")
print("Printing Network parameters.../n")
print("Model's state_dict:")
for param_tensor in r.state_dict():
    #print(param_tensor, "/t", cnn.state_dict()[param_tensor].size())
    print(r.state_dict()[param_tensor].size())
    print(param_tensor)

# Print optimizer's state_dict
print("/nOptimizer's state_dict:")
for var_name in optimizer.state_dict():
    #print(var_name, "/t", optimizer.state_dict()[var_name])
    print(var_name)

print("/nSaving Model...")
torch.save( r.state_dict(), (PATH + "PreTrained_Networks/" + model_pretrained_name) )
print("Saved Model as %s..." % (model_pretrained_name))
#Sample images + Normalized MSE
j = np.random.randint(0,(MBS)-1)
print('j=' + str(j))
noisy_im = inp.cpu().detach().numpy()[j].reshape(185,180)
ground_truth = out.cpu().detach().numpy()[j].reshape(47,180)
im_reconstructed = pred.cpu().detach().numpy()[j].reshape(47,180)
#Plot 
plt.figure()

plt.imshow(ground_truth,cmap = 'gray')
plt.title("Original image")
plt.figure()
plt.imshow(noisy_im,cmap = 'gray')
plt.title("Noisy image")
plt.figure()
plt.imshow(im_reconstructed,cmap = 'gray')
plt.title("Reconstructed image")

#Compute Normalized MSE
n_mse = np.linalg.norm((ground_truth-im_reconstructed)**2,'fro')/np.linalg.norm(ground_truth**2,'fro')
print(n_mse)

#Pass through entire test set
if True:
    MBS = 100
    test_loader = torch.utils.data.DataLoader(testset,batch_size = MBS,shuffle = True)
    with torch.no_grad():
        for data, noisy_data in test_loader:
            inp = Variable(noisy_data.reshape(MBS,noisy_data.shape[1],1).cuda())
            out = Variable(data.reshape(MBS,data.shape[1],1).cuda())
            
            pred = r(inp)
    
    j = np.random.randint(0,99)
    plt.figure()
    
    plt.imshow(inp.cpu().detach().numpy()[j].reshape(185,180),cmap = 'gray')
    plt.title("Noisy image")
    plt.figure()
    plt.imshow(out.cpu().detach().numpy()[j].reshape(185,180),cmap = 'gray')
    plt.title("Original image")
    plt.figure()
    plt.imshow(pred.cpu().detach().numpy()[j].reshape(185,180),cmap = 'gray')
    plt.title("Reconstructed image")
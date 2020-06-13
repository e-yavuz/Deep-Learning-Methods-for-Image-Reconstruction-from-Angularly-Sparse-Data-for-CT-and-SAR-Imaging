import torch.cuda
import torch.utils.data
import torch.nn as nn

import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.autograd import Variable
from architectures import UNet
print("Finished importing...")


"""
============================== CONSTANT VARIABLES =============================
"""
torch.device("cuda")
torch.cuda.synchronize()
EPOCHS = 250  #Run for 2500 epochs
MBS = 25#100 #MBS is 100
LEARNING_RATE = 1E-4 #Learning Rate
PATH = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Python_Sean/codes/Data_Generative_Code/' #path
NUM_IMAGES_TRAIN = 200
NUM_IMAGES_TEST = 50
"""
================================== DATASET ====================================
"""
num_batches = NUM_IMAGES_TRAIN/MBS #Number of training batches (computing loss)
num_batch_test = NUM_IMAGES_TEST/MBS #Number of testing batches (computing loss)

PicklePath = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Data_Generative_Code/PickleDir/'
#Obtain training dataset, initiate train loader. Shuffle the data
DS = 'train_unet_RNNData.pkl' #dataset
with open(PicklePath + DS, 'rb') as f:
    trainset = pickle.load(f)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=MBS,
                                           shuffle=True)

#Obtain testing dataset, initiate test loader. Shuffle the data
DS = 'test_unet_RNNData.pkl'
with open(PicklePath + DS,'rb') as f:
    testset = pickle.load(f)
#print(np.array(testset).shape)
test_loader = torch.utils.data.DataLoader(testset,batch_size = MBS,
                                          shuffle = True)


"""
============================== INSTANTIATE MODEL =============================
"""
#Use BidirectionalRNN with 8 layers. Send to cuda()
cnn= UNet()
cnn.cuda()
criterion = nn.MSELoss() #Loss function is MSE
optimizer = torch.optim.Adam(cnn.parameters(),lr= LEARNING_RATE)

train_loss_plot = []  #Keep track of training loss
test_loss_plot = [] #Keep track of testing loss
model_pretrained_name = "unet_l2_tau1e-3_Ellipsoid_RnnData.pt"

torch.autograd.set_detect_anomaly = False #Prevents memory leaks...

"""
==================================== TRAINING =================================
"""

print("Starting Training...")

for epoch in range(EPOCHS):
    torch.autograd.set_detect_anomaly = False #Prevents leaks
    #Reset loss
    train_loss = 0 
    test_loss = 0
    for i,(data, noisy_data) in enumerate(train_loader):#i,(data,noisy_data) in enumerate(train_loader):
        cnn.train()
        
        #Obtain inputs (nosiy data), output (ground truth), prediction
        inp = Variable(noisy_data.unsqueeze(1).cuda())
            
        out = Variable(data.unsqueeze(1).cuda() )
        pred = cnn(inp).cuda()
        #Obtain loss
        loss = criterion(pred, out)# + lmbda*l1_loss
        train_loss += loss.item()
        loss.backward() #Backpropagate
        optimizer.step() #Step optimizer
        optimizer.zero_grad() #Zero-out gradient
    #Test dataset...ignore for now. Or add into image pre-processing and run
    if True: #change to True if want to run
        with torch.no_grad(): #Do not train when running test set
            for i,(data, noisy_data) in enumerate(test_loader):
                inp = Variable(noisy_data.unsqueeze(1).cuda())
                    
                out = Variable(data.unsqueeze(1).cuda() )
                pred = cnn(inp).cuda()
                #Obtain loss
                loss = criterion(pred, out)# + lmbda*l1_loss
                test_loss += loss.item()
                
                #Saves very last image reconstruction
                
                if epoch%50 == 0:
                    count = 0
                    for j in range(np.shape(pred.cpu().detach().numpy())[0]):
                        count += 1
                        path1 = 'C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Data_Generative_Code/Image_Data/'

                        plt.imsave(path1 + 'Test/' + 'Unet-1/Noisy/' + 'Image_' + str(count) + '_orignial_RNNData.png',
                                   inp.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
                        plt.imsave(path1 + 'Test/' + 'Unet-1/Ground_Truth/' + 'Image_' + str(count) + '_orignial_RNNData.png',
                                   out.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
                        plt.imsave(path1 + 'Test/' + 'Unet-1/Reconstructed/' + 'reconstructed_' + str(count) + '_RNNData.png',
                                   pred.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
                        #"""
    test_loss = test_loss/num_batch_test
    train_loss = train_loss/(num_batches)
    train_loss_plot.append(train_loss) 
    test_loss_plot.append(test_loss)
    if epoch % 1 == 0:
        print("EPOCH: " + str(epoch) + " Training Loss: " + str(train_loss) + "Test Loss: " + str(test_loss))

#Sample images + Normalized MSE
print("Finished Training...")
plt.figure()
plt.plot(train_loss_plot[2:],label = 'Training Loss')
plt.plot(test_loss_plot,label = 'Testing loss')
plt.legend()
train_loss_plot_numpy = np.asarray(train_loss_plot)
test_loss_plot_numpy = np.asarray(test_loss_plot)
np.save(('C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Loss_Data/' + str(model_pretrained_name[:-3]) + '_train.npy'), train_loss_plot_numpy)
np.save(('C:/Users/eyavu/OneDrive/Documents/Programming/Project Workspace/Code and Txt Files/Loss_Data/' + str(model_pretrained_name[:-3]) + '_test.npy'), test_loss_plot_numpy)

print("Printing Network parameters...\n")
print("Model's state_dict:")
for param_tensor in cnn.state_dict():
    #print(param_tensor, "\t", cnn.state_dict()[param_tensor].size())
    #print(cnn.state_dict()[param_tensor].size())
    print(param_tensor)
# Print optimizer's state_dict
print("\nOptimizer's state_dict:")
for var_name in optimizer.state_dict():
    #print(var_name, "\t", optimizer.state_dict()[var_name])
    print(var_name)

print("\nSaving Model...")
torch.save( cnn.state_dict(), (PATH + "PreTrained_Networks/" + model_pretrained_name) )
print("Saved Model as %s..." % (model_pretrained_name))        
j = np.random.randint(0,16)

noisy_im = inp.cpu().detach().numpy()[j].reshape(128,128)
ground_truth = out.cpu().detach().numpy()[j].reshape(128,128)
im_reconstructed = pred.cpu().detach().numpy()[j].reshape(128,128)
#Plot 

plt.figure()

plt.imshow(ground_truth,cmap = 'gray')
plt.title("Ground Truth image")
plt.figure()
plt.imshow(noisy_im,cmap = 'gray')
plt.title("Filtered Backprojection Reconstruction Image")
plt.figure()
plt.imshow(im_reconstructed,cmap = 'gray')
plt.title("U-Net Image")

#Compute Normalized MSE
n_mse = np.linalg.norm((ground_truth-im_reconstructed)**2,'fro')/np.linalg.norm(ground_truth**2,'fro')
print(n_mse)
#Pass through entire test set

if True:
    MBS = 50
    test_loader = torch.utils.data.DataLoader(testset,batch_size = MBS,shuffle = True)
    with torch.no_grad():
        for data, noisy_data in test_loader:
            inp = Variable(noisy_data.unsqueeze(1).cuda())
            
            out = Variable(data.unsqueeze(1).cuda() )
            
            pred = cnn(inp).cuda()
    
    j = np.random.randint(0,50)
    plt.figure()
    
    plt.imshow(inp.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
    plt.title("Noisy image")
    plt.figure()
    plt.imshow(out.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
    plt.title("Original image")
    plt.figure()
    plt.imshow(pred.cpu().detach().numpy()[j].reshape(128,128),cmap = 'gray')
    plt.title("Reconstructed image")
%%Converts existing noisy and ground_truth images into radon transforms for
%RNN, specifically SAR

%User definition of for image number and defining folder location
clear all
close all
clc
 
dir = 'C:\Users\eyavu\OneDrive\Documents\Programming\Project Workspace\Code and Txt Files\Data_Generative_Code\Image_Data\Test\';

    %Code for creating associated radon transforms for each origin image.

GT_Folder = 'Origin_images_set-10_drg3';
GT_Total = 50;
N_Folder = 'Test_images_set-10_drg3_iradon';
N_Total = 50;

mkdir(dir,"Iradon_180");
mkdir(dir,"Iradon_60");


%%
truthpath = [dir , GT_Folder];
addpath(truthpath);
for i = 1:GT_Total
    string_im = sprintf('Image_%i_orignial.png', i)
    im = imread(string_im);   
    im = double(im);
    IR = iradon(im,0:1:179,128);    
        
    imwrite(IR/max(max(IR)),(dir + "Iradon_180\" + string_im) );
end

noise_path = [dir, N_Folder]
addpath(noise_path);
for j = 1:N_Total
    string_im = sprintf('Image_%i_test.png', j);
    im = imread(string_im);
    im = double(im);
    
    Radon_60 = zeros(185,60);
    loop_i = 1;
    loop_j = 1;
    while loop_i < 60
        Radon_60(:,loop_i) = im(:,loop_j);
        loop_i = loop_i + 1;
        loop_j = loop_j + 3;
    end
    
    IR = iradon(Radon_60,0:3:179,128);
    imwrite(IR/max(max(IR)),(dir + "Iradon_60\" + string_im));
end

addpath('C:\Users\eyavu\OneDrive\Documents\Programming\Project Workspace\Code and Txt Files\Python_Sean\codes\Data_Generative_Code\PreTrained_Networks\Paper_Img');
close all
clear all

%Load image and convert to double format

%Original Image - Always Run this (needed for NMSE calculation)
image_name = 'Ellipse_Truth.png';titl = 'Original Phantom';
im_saved = imread(image_name);
im_grnd = double(rgb2gray(im_saved));
%im_g = im_grnd; %Comment out when running other use cases

%image_name = 'Ellipse_Truth180.png';titl = '180 View Inv. Radon Recon.';
%im_saved = imread(image_name);
%im_g = double(im_saved);

%image_name = 'Ellipse_Sparse60.png';titl = '60 View Inv. Radon Recon.';
%im_saved = imread(image_name);
%im_g = double(im_saved);



image_name = 'Ellipse_L2TVRnnData_CNN.png';titl = 'CNN (from 60 view)';
%image_name = 'Ellipse_L2TVRnnData_Unet.png';titl = 'UNet (from 60 view)';
%image_name = 'Ellipse_ReconSinoPaddingL2_RNN.png';titl = 'RNN (from 60 view)';
im_saved = imread(image_name);
im_g = double(rgb2gray(im_saved)); %not for truth
%im_g = iradon(im_g,0:1:179,128); %for RNN


%Read noisy image, convert to double, and then scale by max
%im_2 = rgb2gray(imread('Ellipse_Sparse60.png'));
%im_2_double = double(im_2);
%im_2_double = im_2_double/max(max(im_2_double));
%im_2 = im_2/max(max(im_2));

%Scale Recon or Truth image by max
im = im_g/max(max(im_g));

%Which use case
titl

%EDGE SPREAD
%REGION 1
x2 = 58;
y2 = 65;
%x2 = 62;
%y2 = 77;
Edge_Block = im(x2:x2+6,y2:y2+6);

figure(2);
tst_im=im;
tst_im(x2:x2+6,y2:y2+6)=2*im(x2:x2+6,y2:y2+6);
imshow(tst_im);colorbar;title('Region for Gradient');

%View Region (Remove for True results)
%im(x2:x2+6,y2:y2+6) = 10;
[Gmag,Gdir] = imgradient(Edge_Block);
Gradient_Average = mean(Gmag(:));
Std = std(Gmag(:),1);
Edge_Spread = Gradient_Average/Std


%SNR (RMS Amplitude Division)
%RMS = @(x) sqrt(mean(x.^2));
%SNR = (RMS(im) / RMS(im_2_double)) ^ 2

%CNR
%REGION 1
x_ref = 10:20;
y_ref = 10:20;
x_box = (66:76);
y_box = (65:75);
%x_ref = 40:50;
%y_ref = 80:90;
%x_box = (66:76);
%y_box = (65:75);
%View regions (Remove for true results)
%im(x_ref,y_ref) = 10;
%im(x_box,y_box) = 10;
ref = im(x_ref,y_ref);
ref_m = mean(mean(ref));
ref_var = var(var(ref));

box = im(x_box,y_box);
box_m = mean(mean(box));
box_var = var(var(box));

figure(3);
tst_im=im;
tst_im(x_ref,y_ref)=2*im(x_ref,y_ref)+.5;
tst_im(x_box,y_box)=2*im(x_box,y_box);
imshow(tst_im);colorbar;title('Region for CNR');

CNR = abs(box_m-ref_m)/sqrt(box_var+ref_var)


%NMSE Calculation
    m = 128;
    n = 128;
    tmp = double(zeros(m, m));
    
    %im = ellipseMatrix(m/2, m/2, 40, 60, 0, tmp, 10, 0, 0);
    %im_in = ellipseMatrix(m/2, m/2, 37, 57, 0, tmp, 2, 0, 0);
    im_in = ellipseMatrix(m/2, m/2, 33, 52, 0, tmp, 2, 0, 0);
    indx_in = find (im_in > 0);
    indx_out = find (im_in < 1);
  
im_gscl = im_grnd/max(max(im_grnd));

im_org = im_gscl;
im_org(indx_out) = 0;

im_recon = im;
im_recon(indx_out) = 0;

df = im_recon - im_org;
figure(4);
imagesc(df);title('diff')

nmse_val = sum(sum(df .* df)) / sum(sum(im_org .* im_org))


figure(1);
imshow(im);title(titl);colorbar;
%For radon transforms (Not used here)
%figure;
%imshow(IR/max(max(IR)));colorbar;

%Used for viewing edges, not a metric
%figure(2);
%plot(im(:,110));title('RNN');hold on;
%plot(im(:,111));title('RNN');
%plot(im(:,112));title('RNN');





%figure(3);
%imagesc(im);colorbar;
%figure(4);
%imagesc(im(:,:,1));colorbar;
%figure(5);
%imagesc(im(:,:,2));colorbar;
%figure(6);
%imagesc(im(:,:,3));colorbar;

%figure;
%imagesc(Gmag/max(max(Gmag))); colorbar;

    


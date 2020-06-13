dr = 'C:\Users\eyavu\OneDrive\Documents\Programming\Project Workspace\Code and Txt Files\Python_Sean\codes\Data_Generative_Code\PreTrained_Networks\Paper_Img';
addpath(dr);
close all
clear all
format compact

im_n = 'Ellipse_Truth.png';
im = imread(im_n);
im = double(rgb2gray(im));

range = 0:3:179;
[R,xp] = radon(im,range);
IR = iradon(R,range,128);

%im2 = zeros(size(im));
%fn = find (im > 127);
%im2(fn) = 255;
%im = im2;

figure;
subplot(2,2,1);imagesc(im);colorbar;
subplot(2,2,2);imagesc(IR/max(max(IR)));colorbar;
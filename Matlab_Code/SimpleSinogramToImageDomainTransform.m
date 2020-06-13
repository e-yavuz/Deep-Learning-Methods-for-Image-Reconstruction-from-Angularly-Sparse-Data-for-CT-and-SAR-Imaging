path = 'C:\Users\eyavu\OneDrive\Documents\Programming\Project Workspace\Code and Txt Files\Python_Sean\codes\Data_Generative_Code\PreTrained_Networks\Paper_Img'
imname = 'Ellipse_ReconSinoPaddingL2_RNN.png'; 
range = 0:1:179;
sz = 128;

addpath(path);


im = imread(imname);
im = double(rgb2gray(im));
IR = iradon(im,range,sz);
figure; colormap;
imagesc(IR/max(max(IR)));


%%%%%%%%%%%%%%
close all
clear all
format compact
pth = 'C:\Users\eyavu\OneDrive\Documents\Programming\Project Workspace\Code and Txt Files\Python_Sean\codes\Data_Generative_Code\PreTrained_Networks\Paper_Img'
addpath(pth);
 
imm = imread('Ellipse_180view.png');
sin180 = double(imm);
%double((imm));

imm =imread('Ellipse_60viewPADDED.png');
sin60 = double(imm);

imm =imread('Ellipse_ReconSinoPaddingL2TV_RNN.png');
sinRNN_1 = double(rgb2gray(imm));

imm =imread('Ellipse_ReconSinoPaddingL2_RNN.png');
sinRNN_2 = double(rgb2gray(imm));

imm = imread('Ellipse_ReconSinoPaddingL2L1_RNN.png');
sinRNN_3 = double(rgb2gray(imm));

range = 0:1:179;
sz = 128;


%% Interpolation 
sin60_dwnsmpl=zeros(185,60);
for ii=1:60
   sin60_dwnsmpl(:,ii)= sin180(:,(ii-1)*3 + 1);
end
im = imread('Ellipse_Truth.png');
im = double(rgb2gray(im));
[sin60_dwnsmpl2,xp] = radon(im,0:3:179);
mx1=max(max(sin60_dwnsmpl));
mx2=max(max(sin60_dwnsmpl2));
sin60_dwnsmpl2 = sin60_dwnsmpl2 * mx1 / mx2;


figure(4);
plot(sin60_dwnsmpl(50,:));hold on;
plot(sin60_dwnsmpl2(50,:),'r');hold on;

F = griddedInterpolant(double(sin60_dwnsmpl));
F.Method = 'cubic';
[sx,sy] = size(sin60_dwnsmpl);
xq = (1:sx)';
yq = (0:1/3:(sy-1/3))';
sin60_upsmpl = F({xq,yq});

figure(1);
Img=sin180;subplot(2,2,1);imagesc(Img/max(max(Img)));title('Original');
Img=sin60;subplot(2,2,2);imagesc(Img/max(max(Img)));title('60 View');
Img=sin60_upsmpl;subplot(2,2,3);imagesc(Img/max(max(Img)));title('MatlabInterp');
Img=sinRNN_2;subplot(2,2,4);imagesc(Img/max(max(Img)));title('RNN_L2');


figure(2);
rw=50;
Img=sin180;Img = double((Img));plot(Img(rw,:));hold on;
Img=sin60;Img = double((Img));plot(Img(rw,:),'r');hold on;
Img=sin60_upsmpl;plot(Img(rw,:),'g');hold on;
Img=sinRNN_2;plot(Img(rw,:),'c')


figure(3);
Img=sin180;
subplot(3,2,1);IR = iradon(Img,range,sz);imagesc(IR/max(max(IR)));title('Original');
Img=sin60;
subplot(3,2,2);IR = iradon(Img,range,sz);imagesc(IR/max(max(IR)));title('60 View');
Img=sin60_upsmpl;
subplot(3,2,3);IR = iradon(Img,range,sz);imagesc(IR/max(max(IR)));title('MatlabInterp');
Img=sinRNN_2;
subplot(3,2,4);IR = iradon(Img,range,sz);imagesc(IR/max(max(IR)));title('RNN L2');
Img=sinRNN_1;
subplot(3,2,5);IR = iradon(Img,range,sz);imagesc(IR/max(max(IR)));title('RNN L2TV');
Img=sinRNN_3;
subplot(3,2,6);IR = iradon(Img,range,sz);imagesc(IR/max(max(IR)));title('RNN L2L1');

figure(6);
Img=sinRNN_2;
IR = iradon(Img,range,sz);imagesc(IR/max(max(IR)));title('RNN');colorbar;


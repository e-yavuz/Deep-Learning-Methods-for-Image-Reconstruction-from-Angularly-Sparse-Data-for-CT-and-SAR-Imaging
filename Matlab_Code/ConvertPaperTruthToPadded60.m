path = 'C:\Users\eyavu\OneDrive\Documents\Programming\Project Workspace\Code and Txt Files\Python_Sean\codes\Data_Generative_Code\PreTrained_Networks\Paper_Img';
addpath(path);
im = imread('Ellipse_Truth.png');
im = double(rgb2gray(im));

[R,xp] = radon(im,0:3:179);

padded_undersample = zeros(185,180);
    loop_i = 1;
    loop_j = 1;
    while loop_i < 180
        padded_undersample(:,loop_i) = R(:,loop_j);
        loop_i = loop_i + 3;
        loop_j = loop_j + 1;
    end

[Truth_R,Truth_xp] = radon(im,0:1:179);

%Truth_string = "Ellipse_180view.png";
Truth_string = "SAR_180view.png";
folder_dir = fullfile(path, Truth_string);  %name file relative to that directory

imwrite(Truth_R/max(max(Truth_R)),folder_dir);

Undersample_string = "Ellipse_60viewPADDED.png";
folder_dir = fullfile(path, Undersample_string);  %name file relative to that directory

imwrite(padded_undersample/max(max(padded_undersample)),folder_dir);
    
%%

%NOTE: TAKES ROUGHLY 1.75 SECONDS FOR EACH IMAGE ON INSPIRON 5379 DELL COMPUTER, PLAN ACCORDINGLY

%User definition of for image number and defining folder location
clear all
close all
clc
image_category = input("What type: Train (1) or Test (0)?\n");
if(image_category == 1)
    type = "\Train";
else if(image_category==0)
    type = "\Test";
    end
end
origin_directory = 'C:\Users\eyavu\OneDrive\Documents\Programming\Project Workspace\Code and Txt Files\Data_Generative_Code\Image_Data' + type;
quantity_limit = input("How many images created from data generation?\n");
 

    %Code for creating associated radon transforms for each origin image.
degree = input("What Degree for data generation?\n");

%{
Helps with sorting through each set of images, displays successive set
number automatically
%}
set_number = 1;
valid = false;
while valid == false
    if exist(origin_directory + "\" + sprintf("Origin_images_set-%i_drg%i",set_number, degree))
        set_number = set_number+1;
    else
        valid = true;
    end
end

origin_folder = sprintf("Origin_images_set-%i_drg%i",set_number, degree);


mkdir(origin_directory,origin_folder);

    %Code for creating associated radon transforms for each origin image.
test_folder = sprintf("Test_images_set-%i_drg%i_iradon",set_number, degree)

    %Code for creating associated radon transforms for each origin image.
mkdir(origin_directory,test_folder)

%%
%Code for Ellipse creation 

image_number = 0;
for i = 1:quantity_limit
    image_number = image_number + 1;
    m = 128;
    n = 128;
    tmp = double(zeros(m, m));
    
    im = ellipseMatrix(m/2, m/2, 40, 60, 0, tmp, 10, 0, 0);
    im_in = ellipseMatrix(m/2, m/2, 37, 57, 0, tmp, 2, 0, 0);
    
    indx_out_bone = find( im < .1);
    indx_out_tissue = find( im_in < .1);
    
    indx_in = find (im_in > 0);
    im(indx_in) = im_in(indx_in);
    
    
    
    ytemp1= (1-(randi(25)/25)^2)*25;
    xtemp2=(1-(randi(25)/25)^2)*25;
    flip = randi(2);
    close all;
    
    %For color randomization
    color_arr = [1,3,4,5];
    %Creates random number of ellipses and setting up temporary ellipse structures
    ellp_count = (1+randi(4,1,1));
    ellp_all = zeros(m,m,ellp_count);
    ellp_all_marker = zeros(m,m,ellp_count);
    
       for cnt = 1:ellp_count
           
        flip1 = randi(2);
        flip2 = randi(2);
        if flip1 == 1
            y0 = (m/2)+(1-(randi(25)/25)^2)*25;;
        else
            y0 = (m/2)-(1-(randi(25)/25)^2)*25;;
        end
        if flip2 == 1
            x0 = (m/2)+(1-(randi(25)/25)^2)*25;
        else
            x0 = (m/2)-(1-(randi(25)/25)^2)*25;
        end
        a = randi(25);
        b = randi(25);   
        c = max(a, b);
        theta = 2*pi*rand(1);
        color = color_arr(randi(4,1,1));
        rng('shuffle');
        
        %Adding completed ellipse to ellipse structures
        im_ellipseadd = ellipseMatrix(y0, x0, a, b, theta, tmp, color, 0, 0);
        ellp_all(:,:,cnt) = im_ellipseadd;
        fnd_ellipse_for_marker = find(im_ellipseadd>0);
        tmp2 = zeros(m,m);
        tmp2(fnd_ellipse_for_marker) = 1;
        ellp_all_marker(:,:,cnt) = tmp2;
        
       end

ellp_images = tmp;
ellp_images_sum = sum(ellp_all,3);
ellp_images_marker_sum = sum(ellp_all_marker,3);
fn = find(ellp_images_marker_sum > 0);
ellp_images(fn) = (ellp_images_sum(fn)./ellp_images_marker_sum(fn));
im(fn) = ellp_images(fn);

%{
%Code for checking completed ellipse data structures
figure(1);imagesc(ellp_images_sum);colorbar;
figure(2);imagesc(ellp_images_marker_sum);colorbar;
figure(3);imagesc(ellp_images);colorbar;
%}    
       
%Resetting exterior to prevent 'escaping' ellipses
im(indx_out_tissue)= 10;
im(indx_out_bone)= 0;
%%

%{
%Code for debugging and displaying ellipse
figure(4); colormap gray;
imagesc(im); colorbar;
%}
%%

%Code for writing generated images to files

    %Note: 'im' refers to current image with ellipses already added within

    %Code for creating associated radon transforms for each origin image.
    range = 0:degree:179;
    range2 = 0:1:179;

    %Code for creating radon noisy image 185x180.
    [R,xp] = radon(im,range);
    %IR_noise = iradon(R,range,128);
    %[ResizedR, xpResized] = radon(IR_noise,range2);
    %Code for creating associated radon transforms for each origin image.
    [R2,xp2] = radon(im,range2);
    
    %padded_noise = zeros(185,180);
    %loop_i = 1;
    %loop_j = 1;
    %while loop_i < 180
    %    padded_noise(:,loop_i) = R(:,loop_j);
    %    loop_i = loop_i + 3;
    %    loop_j = loop_j + 1;
    %end

    Original_Image_Return = R2;
    origin_string = sprintf("Image_%i_orignial.png",image_number);
    Write_directory_original = origin_directory + "\" + origin_folder;
    folder_dir = fullfile(Write_directory_original, origin_string);  %name file relative to that directory
    %imwrite(R2/max(max(R2)),folder_dir);
    imwrite(Original_Image_Return/max(max(Original_Image_Return)),folder_dir);
    
    
    SubSampled_Image_Return = padded_noise;
    data_string = sprintf("Image_%i_test.png",image_number);
    Write_directory_IR = origin_directory + "\" + test_folder;
    folder_dir2 = fullfile(Write_directory_IR, data_string);  %name file relative to that directory
    imwrite(SubSampled_Image_Return/max(max(SubSampled_Image_Return)),folder_dir2);
end
Write_directory_original
Write_directory_IR
"Finished Making Images"
%Writing phantom image for comparison
    
    %folder_dir = fullfile(origin_folder, 'Phantom_image.png');  %name file relative to that directory
    %imwrite(phantom(512)/max(max(phantom(512))),folder_dir);

clear all
close all
clc
INITIAL_PATH = pwd;
FOLDER_PATH = ['C:\Users\Hao\Downloads\gg-polo-mixed-terrain-test-images\'];
IMG_PATH = [FOLDER_PATH,'*.jpg'];
image_files = dir(IMG_PATH);
img_num = length(image_files);
image_data = cell(1,2);

GSD = 0.05;% m
ground_size = 100; % m on ground

for img_count = 1:img_num
     image_data{1}{img_count} = image_files(img_count).name;
     result = blur_check([FOLDER_PATH,image_files(img_count).name], GSD, ground_size);   
     image_data{2}{img_count} = result;
     
       disp_str2=[image_files(img_count).name,' ', num2str(image_data{2}{img_count})];
%      disp(disp_str2)
%      disp(' ')
     if result(1)
        disp_str1 = [image_files(img_count).name,' is blurred!'];
        disp(disp_str1)
        disp(' ')
    else
    end    
end

cd(FOLDER_PATH);
% write blur log
imageLogFileID = fopen('imageBlurLog.csv','w+');
fprintf(imageLogFileID,'%s,%s,%s,%s\n', ...
    'file', ...                     
    'blur', ...
    'mean', ...
    'median');
for i = 2:length(img_num)+1
    fprintf(imageLogFileID,'%s,%.2f,%.2f,%.2f\n', ...
    image_files(i-1).name, ...
    image_data{2}{i-1}(1), ...
    image_data{2}{i-1}(2), ...
    image_data{2}{i-1}(3));
end

cd(INITIAL_PATH);


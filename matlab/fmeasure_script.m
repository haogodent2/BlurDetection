clc
clear all
close all

%img import
I1 = imread('no blur.jpg');
I_gray1 = rgb2gray(I1);
I2 = imread('slight blur.jpg');
I_gray2 = rgb2gray(I2);
[img_width, img_height] = size(I1);

%declarations
all_methods = {'ACMO','BREN','CONT','CURV','DCTE','DCTR','GDER','GLVA',...
    'GLLV','GLVN','GRAE','GRAT','GRAS', 'HELM', 'HISE', 'HISR','LAPE',...
    'LAPM','LAPV','LAPD','SFIL','SFRQ','TENG','TENV','VOLA','WAVS',...
    'WAVV','WAVR'};
centeral_difference = [-0.5,0,0.5]; % derivative approximation mask
second_central_difference = [1,-2,1]; % 2nd derivative approximation mask

% VARIABLES
GSD_m = 0.04;% m
patch_size_m = 25; % m
threshold = [0.3]; % for edge detection
methods = {'LAPE','LAPM','LAPV','LAPD'}; % to use



ROI_size = (patch_size_m/GSD_m);
ROI = [floor((img_width-ROI_size)/2),floor((img_width-ROI_size)/2),ROI_size,ROI_size]; %[xo yo width heigth]
%visualize raw patch
figure
subplot(2,1,1);
title('not blurred')
imshow(I_gray1(ROI(1):ROI(1)+ROI(3),ROI(2):ROI(2)+ROI(4)));
subplot(2,1,2);
title('blurred')
imshow(I_gray2(ROI(1):ROI(1)+ROI(3),ROI(2):ROI(2)+ROI(4)));

%edge detection
I_gray1_edge = edge(I_gray1(ROI(1):ROI(1)+ROI(3),ROI(2):ROI(2)+ROI(4)),'canny',threshold);
I_gray2_edge = edge(I_gray2(ROI(1):ROI(1)+ROI(3),ROI(2):ROI(2)+ROI(4)),'canny',threshold);

%visualize raw patch
figure
subplot(2,1,1);
title('not blurred')
imshow(I_gray1_edge);
subplot(2,1,2);
title('blurred')
imshow(I_gray2_edge);



%%
num = length(methods);
FM1 = zeros(1,num);
FM2 = zeros(1,num);
for count = 1:num
    FM1(count) = fmeasure(I_gray1, methods{count},ROI);
    FM2(count) = fmeasure(I_gray2, methods{count},ROI);
end

% use only values < 1000
FM1_filtered_index = find(FM1<100);
FM1_filtered = FM1(FM1_filtered_index);

FM2_filtered_index = find(FM2<100);
FM2_filtered = FM2(FM2_filtered_index);
% subplot(2,2,3)
% title('not blurred')
% plot(FM1_filtered,'*r')
% ylim([0,100]);
% subplot(2,2,4)
% title('blurred')
% plot(FM2_filtered,'*b')
% ylim([0,100]);

figure
hold on
plot(FM1_filtered,'*r')
plot(FM2_filtered,'*b')
ylim([0,100]);
legend('not blurred','blurred')


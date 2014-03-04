function [blur_result]=blur_check(IMG_PATH,GSD,ground_size)

I_original = imread(IMG_PATH);
I_gray = rgb2gray(I_original);
[IMG_WIDTH, IMG_HEIGHT] = size(I_gray);



ROI_size = (ground_size/GSD);
ROI = [floor((IMG_WIDTH-ROI_size)/2),floor((IMG_HEIGHT-ROI_size)/2),ROI_size,ROI_size]; %[xo yo width heigth]

I_gray_cropped = I_gray(ROI(1):ROI(1)+ROI(3),ROI(2):ROI(2)+ROI(4));
% figure
% subplot(1,2,1)
% title('gray')
% imshow(I_gray_cropped);


% detect whether there are strong edges present, building?
% if no strong edges present, must be ground, process accordingly

threshold = [0.03]; % for edge detection
I_gray_edge = edge(I_gray_cropped,'sobel',threshold);
% subplot(1,2,2)
% title('edge')
% imshow(I_gray_edge);

[Gmag, Gdir] = imgradient(I_gray_edge);

Gdir(Gdir==0)=[];
Gdir(Gdir<0)= Gdir(Gdir<0)+180;
Gdir(Gdir==180)= 0;

nelement = hist(Gdir,[0 45 90]);
val_spread = std(nelement);
% if val_spread > 1000
% directional, not just ground patch
edges = edge_width_measure(I_gray_cropped,I_gray_edge);
nelements = hist(edges(:,3));
thickness_mean = mean(edges(:,3));
thickness_median = median(edges(:,3));

if thickness_median > 15
    blur_result = [1,thickness_mean ,thickness_median];
else
    blur_result = [0,thickness_mean ,thickness_median];
end






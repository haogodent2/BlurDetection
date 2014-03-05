'''
Created on Mar 3, 2014
for testing different edge detection methods
@author: Hao
'''
from glob import glob
from os import path
import shutil

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans2

import blurdetector

img_names = glob('../../images/*.jpg')
dummy_img_name= path.basename(img_names[0])
img_dir = img_names[0].rstrip(dummy_img_name)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)

ROI_SIZE_INPUT = 50 # meter
GSD = .03 # meter/pixel
ROI_LENGTH = ROI_SIZE_INPUT//GSD
print ROI_LENGTH
BLUR_WIDTH_THRESHOLD = 13
widths = []

for name in img_names:
    src = name
    base_name = path.basename(name)
    img = cv2.imread(name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    ROW_COUNT, COL_COUNT = img.shape
    ROW_CENTER= ROW_COUNT//2
    COL_CENTER = COL_COUNT//2
    
    ROW_BEGIN = ROW_CENTER-(ROI_LENGTH//2)
    ROW_END = ROW_CENTER+(ROI_LENGTH//2)
    COL_BEGIN = COL_CENTER-(ROI_LENGTH//2)
    COL_END = COL_CENTER+(ROI_LENGTH//2)
    ROI = (ROW_BEGIN, ROW_END, COL_BEGIN, COL_END)

    # try resize vs taking a piece
    #img_small = cv2.resize(img,(col_number, row_number))
    
    img_ROI = img[ROI[0]:ROI[1],ROI[2]:ROI[3]]
    #img_small = img
    
    # image conditioning
    # denoise so edges can be counted better
    #img_ROI_denoised = cv2.fastNlMeansDenoising(img_ROI) # denoising! 
    img_ROI_denoised = cv2.medianBlur(img_ROI,3)
    
    #contrast equalization so more edges can be found   
    img_ROI_denoised_equalized = cv2.equalizeHist(img_ROI_denoised)
    ### image blurring to remove noise from edge detection
    ksize = 3
    sigma_x = 1
    img_ROI_denoised_equalized_smoothed = cv2.GaussianBlur(img_ROI_denoised_equalized,(ksize,ksize),sigma_x)
    
    ### edge detect
    
    # defining canny thresholds
    img_median = np.median(img_ROI_denoised_equalized_smoothed)
    high_threshold = 1.33*img_median
    low_threshold = 0.66*img_median
    '''
    img_max = img_ROI_denoised_equalized_smoothed.max()
    high_threshold = 0.8*img_max
    low_threshold = 0.5*img_max
    '''
    edges = cv2.Canny(img_ROI_denoised_equalized_smoothed, low_threshold, high_threshold)

    edge_elements = blurdetector.measure_edge_width(img_ROI_denoised_equalized, edges)
    
    num_of_edges = len(edge_elements)

    if num_of_edges != 0:
        # edge width metrics
        #avg_widths = (np.mean(edge_elements[:,2]))
        #avg_widths = (np.median(edge_elements[:,2]))
        avg_widths = (sum(edge_elements[:,2])/num_of_edges)
        
        print name + ' width, ' + str(avg_widths)
        widths.append(avg_widths)    
    else:
        print 'no edges found!'
        dst = img_dir + 'unknown/' + base_name
        shutil.move(src,dst)
        
    if avg_widths >= BLUR_WIDTH_THRESHOLD:
        dst = img_dir + 'blurred/' + base_name
        shutil.move(src,dst)

    
    
    img_combined = np.hstack((img_ROI_denoised_equalized,edges))
    img_resized = cv2.resize(img_combined,(1200,700))
    cv2.imshow('image',img_resized)
    cv2.waitKey(5000)

cv2.destroyAllWindows()

# setting blur metric threshold
'''
#kmeans
np_widths = np.array(widths)
clusters = kmeans2(np_widths, 2)
line_y_val = np.mean((clusters[0][0], clusters[0][1]))
'''
#hard coded
line_y_val = 13

fig1 = plt.figure('1')
ax1= fig1.gca()
plt.subplot(211)
# img index based
plt.plot(widths,'r*')
plt.axhline(line_y_val, xmin=0, xmax=1)

plt.subplot(212)
#hist
plt.hist(widths)
plt.show()

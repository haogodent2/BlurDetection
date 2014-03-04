'''
Created on Mar 3, 2014
for testing different edge detection methods
@author: Hao
'''
import glob
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans2
import blurdetector

img_names = glob.glob('../../images/*.jpg')
cv2.namedWindow('image',cv2.WINDOW_NORMAL)

ROI_SIZE_INPUT = 25 # meter
GSD = .03 # meter/pixel
ROI_LENGTH = ROI_SIZE_INPUT//GSD
print ROI_LENGTH
widths = []
for name in img_names:
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
    
    img_small = img[ROI[0]:ROI[1],ROI[2]:ROI[3]]
    #img_small = img
    
    # image conditioning
    img_small = cv2.fastNlMeansDenoising(img_small) # denoising!     
    img_small = cv2.equalizeHist(img_small) #contrast equalization
    
    ksize = 3
    sigma_x = 1
    ### smoothing
    img_smoothed = cv2.GaussianBlur(img_small,(ksize,ksize),sigma_x)
    
    ### canny
    # threshold setting
    '''
    img_median = np.median(img_small)
    high_threshold = 1.33*img_median
    low_threshold = 0.66*img_median
    '''
    
    img_max = img_small.max()
    high_threshold = 0.8*img_max
    low_threshold = 0.5*img_max
    
    edges = cv2.Canny(img_smoothed, low_threshold, high_threshold)
    
    '''
    ### condition edges to remove noise
    kernel = np.ones((2,2),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    ### filtering?
    '''
    num_of_edges = len(edges.nonzero()[1])
    edge_elements = blurdetector.measure_edge_width(img_smoothed, edges)
    if num_of_edges != 0:
        avg_widths = (np.mean(edge_elements[:,2]))
        #avg_widths = (np.median(edge_elements[:,2]))
        #avg_widths = (sum(edge_elements[:,2])/num_of_edges)
        
        print name + ' width, ' + str(avg_widths)
        widths.append(avg_widths)    
    else:
        print 'no edges found!'
    
    img_combined = np.hstack((img_small,edges))
    img_resized = cv2.resize(img_combined,(1200,700))
    
    cv2.imshow('image',img_resized)
    cv2.waitKey(100)

cv2.destroyAllWindows()

# clustering threshold

np_widths = np.array(widths)
clusters = kmeans2(np_widths, 2)
line_y_val = np.mean((clusters[0][0], clusters[0][1]))

#hard threshold
#line_y_val = 8
fig = plt.figure()
ax= fig.gca()
plt.plot(widths,'r*')
plt.axhline(line_y_val, xmin=0, xmax=1)
plt.show()

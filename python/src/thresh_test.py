'''
Created on Feb 19, 2014

for testing threshold values

@author: Hao
'''

import cv2
import numpy as np

high_threshold = 70
low_threshold = 30

def nothing(x):
    pass

gray_image = cv2.imread("2013-10-03-3097.jpg",cv2.CV_LOAD_IMAGE_GRAYSCALE)
gray_image = gray_image[1000:1500,5000:5500]

cv2.namedWindow("original",cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("edges",cv2.WINDOW_AUTOSIZE)

cv2.createTrackbar('high_thresh','edges', 0, 1020, nothing)
cv2.createTrackbar('low_thresh','edges', 0, 1020, nothing)

cv2.imshow("original",gray_image)

while(True):
    high_threshold = cv2.getTrackbarPos("high_thresh","edges")
    low_threshold = cv2.getTrackbarPos("low_thresh","edges")
    
    # calculate canny thresholds
    '''
    img_median = np.median(gray_image)
    high_threshold = 1.33*img_median
    low_threshold = 0.66*img_median
    '''

    edges = cv2.Canny(gray_image,high_threshold,low_threshold, apertureSize=3)
    cv2.imshow("edges",edges)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cv2.destroyAllWindows()
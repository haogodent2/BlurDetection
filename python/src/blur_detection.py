'''
Created on Mar 3, 2014
for testing different edge detection methods
@author: Hao
'''
import glob
from os import path
import shutil
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.vq import kmeans2


def blur_detect(img, AREA_SIZE=50, GSD=0.03, BLUR_WIDTH_THRESHOLD=13.0, thresh_type='static', debug_flag=False):
    '''This function detects whether an image is blurry or not based on the width of edges found in the examined area
            input: 
                img:np.array    source image
                AREA_SIZE:int    size of the square patch to examine in meters
                GSD:float    GSD in meters
                BLUR_WIDTH_THRESHOLD:float    number of pixels beyond which an image is deemed blurry
                thresh_type: str    'static' or 'kmeans'. this determines whether provided BLUR_WIDTH_THRESHOLD should be used or one should be calculated using kmeans clustering
        
            output:
                blur_metric = [width, BLUR_WIDTH_THRESHOLD, blurred]
                width:float    weighted width of edges
                BLUR_WIDTH_THRESHOLD:float either the same as input or value calculated using kmeans depending on thresh_type flag
                blurred: bool True if blurred False not blurred None if uknown
    '''

    blurred = None
    ROI_WIDTH = AREA_SIZE//GSD
    if debug_flag:
        print 'ROI width pixel count = ' + str(ROI_WIDTH)

    ROW_COUNT, COL_COUNT = img.shape
    ROW_CENTER= ROW_COUNT//2
    COL_CENTER = COL_COUNT//2
    
    ROW_BEGIN = ROW_CENTER-(ROI_WIDTH//2)
    ROW_END = ROW_CENTER+(ROI_WIDTH//2)
    COL_BEGIN = COL_CENTER-(ROI_WIDTH//2)
    COL_END = COL_CENTER+(ROI_WIDTH//2)
    ROI = (ROW_BEGIN, ROW_END, COL_BEGIN, COL_END)

    # try resize vs taking a piece
    img_ROI = img[ROI[0]:ROI[1],ROI[2]:ROI[3]]
    
    ### image conditioning
    # denoise so edges can be counted better, median seems to introduce less error at this point
    #img_ROI_denoised = cv2.fastNlMeansDenoising(img_ROI) # denoising! 
    img_ROI_denoised = cv2.medianBlur(img_ROI,3)
    
    ###contrast equalization so more edges can be found   
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
    ''' at this point median seems to do a good job, max is just an alternative
    img_max = img_ROI_denoised_equalized_smoothed.max()
    high_threshold = 0.8*img_max
    low_threshold = 0.5*img_max
    '''
    edges = cv2.Canny(img_ROI_denoised_equalized_smoothed, low_threshold, high_threshold)

    edge_elements = measure_edge_width(img_ROI_denoised_equalized, edges)
    
    num_of_edges = len(edge_elements)

    if num_of_edges != 0:
        # edge width metrics
        avg_width = (np.mean(edge_elements[:,2]))
        #avg_width = (np.median(edge_elements[:,2]))
    else:
        avg_width = None
        if debug_flag:
            print 'no edges found!'
            
    cv2.namedWindow('preview',cv2.WINDOW_NORMAL)
    img_combined = np.hstack((img_ROI_denoised_equalized,edges))
    img_resized = cv2.resize(img_combined,(1200,700))
    cv2.imshow('preview',img_resized)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    # setting blur metric threshold
    
    if avg_width is None:
        pass
    elif avg_width >= BLUR_WIDTH_THRESHOLD:
        blurred = True
    else:
        blurred = False
    
    return [avg_width, BLUR_WIDTH_THRESHOLD, blurred]


def measure_edge_width(img,edge_matrix):
    '''  
    input: gray img, edge matrix
    output: list of lists with each list containing [row, col, edge_width]
    '''    
    padding = 15 # only use edges inside of padding to avoid noise on the boarder of the picture
    # calculate x derivative using sobel
    gradx = cv2.Sobel(img,cv2.CV_32F,1,0)
    grady = cv2.Sobel(img,cv2.CV_32F,0,1)
    img_height,img_width = img.shape
    
    row_padded,col_padded = edge_matrix[padding:-1-padding,padding:-1-padding].nonzero()
    row_non_zero = row_padded+padding
    col_non_zero = col_padded+padding
    num_of_edges = len(row_non_zero)
    #print num_of_edges
    
    edges = np.empty([num_of_edges,3])
    edges[:,0] = row_non_zero
    edges[:,1] = col_non_zero
    
    # thin down the number of edges to 1000-2000
    if num_of_edges > 2000:
        step = num_of_edges//1000
        edges = edges[0:-1:step]
    
    #print len(edges)
    edge_pixel = 1

    for idx in range(0,len(edges)):
        current_row = edges[idx,0]
        current_col = edges[idx,1]
        
        #### X direction
        edge_xdot_sign = math.copysign(1,gradx[current_row,current_col])
        
        left_found = False
        left_increment = 1
        left_col = current_col-left_increment
        
        while not left_found:
            if math.copysign(1,gradx[current_row,left_col]) != edge_xdot_sign:
                left_found = True
            else:
                left_increment += 1
                left_col -= left_increment
                if left_col <=1 or left_increment >= 20:
                    break
                
        right_found = False
        right_increment = 1
        right_col = current_col+right_increment
        
        while not right_found:
            if math.copysign(1,gradx[current_row,right_col]) != edge_xdot_sign:
                right_found = True
            else:
                right_increment += 1
                right_col += right_increment
                if right_col >= img_width-2 or right_increment >= 20:
                    break
            
            
        ####  Y direction    
        edge_ydot_sign = math.copysign(1,grady[current_row,current_col])
        
        up_found = False
        up_increment = 1
        up_row = current_row-up_increment
        
        while not up_found:
            if math.copysign(1,grady[up_row,current_col]) != edge_ydot_sign:
                up_found = True
            else:
                up_increment += 1
                up_row -= up_increment
                if up_row <=1 or up_increment >= 20:
                    break
                
        down_found = False
        down_increment = 1
        down_row = current_row + down_increment
        
        while not down_found:
            if math.copysign(1,grady[down_row,current_col]) != edge_ydot_sign:
                down_found = True
            else:
                down_increment += 1
                down_row += down_increment
                if down_row >= img_height-2 or down_increment >= 20:
                    break
                
        edge_x_pixel =     left_increment + edge_pixel + right_increment    
        edge_y_pixel = up_increment + edge_pixel + down_increment
            
        edges[idx,2] = math.sqrt(pow(edge_x_pixel,2)+pow(edge_y_pixel,2))
        
    return edges        

    



def main():
    
    
    img_dir = '../../images/'
    BLUR_WIDTH_THRESHOLD_INPUT = 13
    img_paths_list = glob.glob(img_dir+'*.jpg')
    if not img_paths_list:
        print 'did you forget to provide the images???'
    img_stat = []
    widths = []
    for img_path in img_paths_list:
        blur_metric = [img_path]
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        blur_output = blur_detect(img , BLUR_WIDTH_THRESHOLD= BLUR_WIDTH_THRESHOLD_INPUT)
        blur_metric.append(blur_output)
        img_stat.append(blur_metric)
        widths.append(blur_output[0])
        print blur_metric[0] + ' edge width = ' + str(blur_output[0])
       
       
    for img_item in img_stat:
        src = img_item[0]
        base_name= path.basename(src) 
        
        if img_item[1][2]:
            dst = img_dir + 'blurred/' + base_name
            if not os.path.isdir(dst.rstrip(base_name)):
                os.makedir(dst.rstrip(base_name))
            shutil.move(src,dst)
            
        elif img_item[1][2] == None:
            dst = img_dir + 'unknown/' + base_name
            if not os.path.isdir(dst.rstrip(base_name)):
                os.makedir(dst.rstrip(base_name))
            shutil.move(src,dst)

    plt.subplot(211)
    # img index based
    plt.plot(widths,'r*')
    plt.axhline(BLUR_WIDTH_THRESHOLD_INPUT)
    
    plt.subplot(212)
    #hist
    plt.hist(widths)
    plt.show()
        
        
        
if __name__ == '__main__': main()

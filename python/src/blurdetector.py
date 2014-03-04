#! /opt/local/bin/python
''' Blur Detector module


'''
import glob
import math

import cv2 
from matplotlib import pyplot as plt
import numpy as np


def blur_measure(img_path):
	#parms for ROI
	ROI_SIZE_INPUT = 50 # meter
	GSD = .034 # meter/pixel

	ROI_WIDTH = ROI_SIZE_INPUT//GSD
	ROI = [0, 0, ROI_WIDTH, ROI_WIDTH] #[row, col, width, height]
	
	# img resize param for imshowing
	if ROI_WIDTH > 500:
		imshow_size = (500,500)
	else:
		imshow_size = (ROI_WIDTH,ROI_WIDTH)

	img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	cropped_img = img[ROI[0]:ROI[0]+ROI[-1], ROI[1]:ROI[1]+ROI[-2]]
	resized_img = cv2.resize(cropped_img,imshow_size)
	
	#smooth image 
	ksize = 5
	sigma_x = 3
	cropped_img = cv2.GaussianBlur(cropped_img,(ksize,ksize),sigma_x)
	
	# calculate canny thresholds
	img_median = np.median(cropped_img)
	high_threshold = 1.33*img_median
	low_threshold = 0.66*img_median
	edge_matrix = cv2.Canny(cropped_img, low_threshold, high_threshold)
	
	'''
	#dilate and erode to remove weak edges
	dilate_kernel = np.ones((4,4),np.uint8)
	erode_kernel = np.ones((3,3),np.uint8)
	dilated_edges = cv2.dilate(edges,dilate_kernel,iterations=1)
	eroded_edges = cv2.erode(dilated_edges,erode_kernel,iterations=1)
	'''
	resized_edges = cv2.resize(edge_matrix,imshow_size)
	edges = measure_edge_width(cropped_img, edge_matrix)
	
	edge_widths = edges[:,2]
	'''
	plt.hist(edge_widths)
	plt.show()	
	'''
	if len(edge_widths) == 0:
		return_val = (edges,0,resized_img,resized_edges)
	else:
		blur_metric = sum(edge_widths)/len(edge_widths)
		return_val = (edges,blur_metric,resized_img,resized_edges)
	return return_val
	
	'''
	plt.plot(resized_gradx[imshow_size[1]//2, :])	  # plots 1th row 
	plt.show()
	'''



def measure_edge_width(img,edge_matrix):
	'''  
	input: gray img, edge matrix
	output: list of lists with each list containing [row, col, edge_width]
	'''	
	padding = 15; # only use edges inside of padding to avoid noise on the boarder of the picture
	
	# calculate x derivative using sobel
	gradx = cv2.Sobel(img,cv2.CV_32F,1,0)
	grady = cv2.Sobel(img,cv2.CV_32F,0,1)
	img_height,img_width = img.shape
	
	row_padded,col_padded = edge_matrix[padding:-1-padding,padding:-1-padding].nonzero()
	row_non_zero = row_padded+padding
	col_non_zero = col_padded+padding
	
	edges = np.empty([len(row_non_zero),3])
	edges[:,0] = row_non_zero
	edges[:,1] = col_non_zero
	
	for idx in range(0,len(edges)):
		current_row = edges[idx,0]
		current_col = edges[idx,1]
		edge_xdot_sign = math.copysign(1,gradx[current_row,current_col])
		edge_pixel = 1
		left_found = False
		left_increment = 1
		left_col = current_col-left_increment
		
		while not left_found:
			if math.copysign(1,gradx[current_row,left_col]) != edge_xdot_sign:
				left_found = True
			else:
				left_increment += 1
				left_col -= left_increment
				if left_col <=1:
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
				if right_col >= img_width-2:
					break
		edges[idx,2] = left_increment + edge_pixel + right_increment		
	return edges		
			

def nothing(arg):
	pass	

def main():
	img_name = glob.glob('../images/*.jpg')
	blur_metrics = np.zeros([len(img_name),1])
	num_of_imgs = float(len(img_name))
	cv2.namedWindow("test",cv2.WINDOW_NORMAL)
	progress= 0
	
	for img_idx in range(0,len(img_name)):
		'''
		current_progress = int(10*((((img_idx)/num_of_imgs)*100)//10))
		if progress != current_progress:
			print  '%d percent complete' %current_progress
			progress = current_progress		
		'''
		
		blur_vals = blur_measure(img_name[img_idx])
		blur_metrics[img_idx,0] = blur_vals[1]
		print str(blur_vals[1]) + ' :: ' + img_name[img_idx]
		
	#img_plot = np.hstack((blur_vals[2],blur_vals[3]))
	#cv2.imshow("test", img_plot)
	plt.plot(blur_metrics,'r')
	plt.show()

	while(True):
		k = cv2.waitKey(10) & 0xff
		if k == 27:
			break
	cv2.destroyAllWindows()
	

if __name__ == "__main__": main()


	
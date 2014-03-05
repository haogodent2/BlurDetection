#! /opt/local/bin/python
''' Blur Detector module


'''
import math

import cv2 
import numpy as np

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
				
		edge_x_pixel = 	left_increment + edge_pixel + right_increment	
		edge_y_pixel = up_increment + edge_pixel + down_increment
			
		edges[idx,2] = math.sqrt(pow(edge_x_pixel,2)+pow(edge_y_pixel,2))
		
	return edges		

	
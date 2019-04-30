from keypointDetect import*
import math
import numpy as np


## This function takes a rectangular window around the keypoints
# row , col are the pixel location of the keypoint
def get_window(window_size, row, col):
	window = []
	lo = math.floor(window_size/2)
	hi = window_size-lo
	for i in range(-lo, hi):
		for j in range(-lo,hi):
			curr_row = row+i
			curr_col = col+j
			window.append((curr_row,curr_col))
	return window


##this function calculates the gradient of magnitude and orientation of a given window around a keypoint
def cal_grad_mag_orient(im,window):
	mag = np.zeros((len(window),))
	orient = np.zeros((len(window),))
	for i in range(len(window)):
		row , col =window[i]
		mag[i]= np.sqrt((im[row,col+1]-im[row,col-1])**2+(im[row+1,col]-im[row-1,col])**2)
		orient[i]= np.arctan((im[row+1,col]-im[row-1,col])/(im[row,col+1]-im[row,col-1]))
	return mag, orient





## This function calculates the gradient and magnitude
## of pixels around each keypoints
def grad_mag_orient(im , locsDoG):
	#this list stores the location of keypoints and their conrresponding surrounding gradient
	#magnitude and orientaions
	result = []

	window_size = 3
	lo = -math.floor(window_size/2)
	hi = window_size-lo
	key_point_num= locsDoG.shape[0]
	H, W = im.shape

	#loop through key points
	for i in range(key_point_num):
		(row , col , level) = locsDoG[i,:]
		if row+lo > 0 and row + hi< H and col+lo>0 and col+hi< W:
			#take a window around the keypoint
			window = get_window(window_size,int(row),int(col))
			#calculate the gradient of magnitude and orientation of the window
			mag,orient = cal_grad_mag_orient(im,window)
			#stores result
			result.append((row,col,mag,orient))

	return result


# my_grad_mag_orientation: Takes in image and calculate the gradient magnitude
#						   and orientation for each pixel
# return: magnitudes->HxWxL matrix with gradient magnitudes
#		  orientation->HxWxL matrix with gradient orientations, range is (0,36)
def my_grad_mag_orientation(locsDoG, gauss_pyramid):
	H = gauss_pyramid.shape[0]
	W = gauss_pyramid.shape[1]
	level_num = gauss_pyramid.shape[2]

	magnitudes = np.zeros((gauss_pyramid.shape[0], gauss_pyramid.shape[1], level_num))
	orientations = np.zeros((gauss_pyramid.shape[0], gauss_pyramid.shape[1], level_num))

	for l in range(0, level_num):
		im = gauss_pyramid[:,:,l]
		for r in range(0, H-1):
			for c in range(0, W-1):
				magnitudes[r,c,l] = ((im[r+1,c]-im[r-1,c])**2 + 
									(im[r,c+1]-im[r,c-1])**2) ** 0.5
				orientations[r,c,l] = (36 / (2*np.pi)) * np.arctan2((im[r+1,c]-im[r-1,c]), 
																	(im[r,c+1]-im[r,c-1]))
	return magnitudes, orientations





if __name__ == '__main__':
    im = cv2.imread('../data/model_chickenbroth.jpg')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    locsDoG, gauss_pyramid= DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12)

    # result = grad_mag_orient(gray, locsDoG)
    WINDOW_SIZE = 5
	# get the gradient magnitude and orientation for each level of the gaussien pyramid
    gradient_mag, gradient_orientation = my_grad_mag_orientation(locsDoG, gauss_pyramid)
    

# SIFT.py: SIFT descriptor implementation
# 16-720 Computer Vision @ Carnegie Mellon University
# Author: Henry Zhang <hzhang0407@gmail.com>
#		  Stefan Zhu <>

from keypointDetect import*
import math
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

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
#		  orientation->HxWxL matrix with gradient orientations, range is (0,8)
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
				orientations[r,c,l] = (8 / (2*np.pi)) * np.arctan2((im[r+1,c]-im[r-1,c]), 
																	(im[r,c+1]-im[r,c-1]))
	return magnitudes, orientations


# compute_desciptors: Takes in locsDoG (local extremas), gradient_mag, gradient_ori
# return: Nx32 matrix with each row to be a 32 element feature vector as the descriptor of the keypoint.
# 		  Feature is computed using 8x8 window, turns to 2x2 histogram descriptor array, each with 8 
#         orientation bars -> 2x2x8 = 32 elements
def compute_descriptor(locsDoG, gradient_mag, gradient_ori):
	# parameters
	N = locsDoG.shape[0]
	H = gradient_mag.shape[0]
	W = gradient_mag.shape[1]
	WINDOW_SIZE = 8
	descriptors = np.zeros((N,32))
	
	# go through the local extremas
	for i in range(0, N):
		locs_extr = locsDoG[i,:]
		row = int(locs_extr[1])
		col = int(locs_extr[0])
		level_index = int(locs_extr[2])
		descriptor = np.zeros(32)

		# calculate weight for points in the window
		gauss_window = multivariate_normal(mean=(row,col), cov=(1.5*WINDOW_SIZE)**2)
		for j in range(-4,4):
			for k in range(-4,4):
				p_row = row + j
				p_col = col + k
				if (p_row < 0 or p_row >= H or p_col < 0 or p_col >= W):
					continue
				weight = gradient_mag[p_row, p_col, level_index] * gauss_window.pdf([p_row, p_col])
				ori_index = np.clip(gradient_ori[p_row, p_col, level_index], 0, 7).astype(int)

				# assign weight to descriptor
				if (j >= -4 and j < 0 and k >= -4 and k < 0):
					# upper-left section
					section_index = 0
				elif (j >= 0 and k >= -4 and k < 0):
					# lower-left section
					section_index = 1
				elif (j >= -4 and j < 0 and k >= 0):
					# upper-right section
					section_index = 2
				else:
					# lower-right section
					section_index = 3
				starting_index = section_index * 8
				descriptor[starting_index + ori_index] += weight
				# normalize the vector
				descriptor = descriptor / np.linalg.norm(descriptor)
				descriptor = np.clip(descriptor, 0, 0.2)
				descriptor = descriptor / np.linalg.norm(descriptor)

		# append the descriptor to the result matrix
		descriptors[i, :] = descriptor
	
	return descriptors


def siftLite(im):
	# define gaussian pyramid levels
	levels = [-1,0,1,2,3,4]
	k = np.sqrt(2)

	# find local extremas and gaussian pyramid of the image
	locsDoG, gauss_pyramid= DoGdetector(im, sigma0=1, k=k, levels=levels, 
										th_contrast=0.03, th_r=12)

	# result = grad_mag_orient(gray, locsDoG)
	# get the gradient magnitude and orientation for each level of the gaussien pyramid
	gradient_mag, gradient_ori = my_grad_mag_orientation(locsDoG, gauss_pyramid)

	# compute the descriptor of keypoints in locsDoG
	descriptors = compute_descriptor(locsDoG, gradient_mag, gradient_ori)

	return locsDoG, descriptors


def siftMatch(desc1, desc2, ratio=0.7):
	
	D = cdist(np.float32(desc1), np.float32(desc2), metric='euclidean')
    # find smallest distance
	ix2 = np.argmin(D, axis=1)
	d1 = D.min(1)
	# find second smallest distance
	d12 = np.partition(D, 2, axis=1)[:,0:2]
	d2 = d12.max(1)
	r = d1/(d2+1e-10)
	is_discr = r<ratio
	ix2 = ix2[is_discr]
	ix1 = np.arange(D.shape[0])[is_discr]

	matches = np.stack((ix1,ix2), axis=-1)
	return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()
    # plt.savefig()


if __name__ == '__main__':
	im1 = cv2.imread('../data/pf_desk.jpg')
	im2 = cv2.imread('../data/pf_floor.jpg')

	locs1, desc1 = siftLite(im1)
	locs2, desc2 = siftLite(im2)

	matches = siftMatch(desc1, desc2)
	plotMatches(im1, im2, matches, locs1, locs2)
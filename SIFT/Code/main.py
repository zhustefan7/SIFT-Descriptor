import numpy as np
import cv2
import keypointDetect

image=cv2.imread('../data/model_chickenbroth.jpg')
# image = cv2.imread('../data/room_l.jpeg')
# image = cv2.imread('../data/incline_L.png')
im_pyramid=keypointDetect.createGaussianPyramid(image, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4])

DoG_pyramid, DoG_levels=keypointDetect.createDoGPyramid(im_pyramid, levels=[-1,0,1,2,3,4])
principal_curvature=keypointDetect.computePrincipalCurvature(DoG_pyramid)
# keypointDetect.getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
#         th_contrast=0.03, th_r=12)


locsDoG, gauss_pyramid= keypointDetect.DoGdetector(image, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12)



def visualize_keypoints(locsDoG,img):
	length,channels=locsDoG.shape
	# print(type(img[1,1]))
	for i in range(length):
		curr_row=int(locsDoG[i,1])
		curr_col=int(locsDoG[i,0])
		# print(curr_row)
		# print(img[curr_row,curr_col,0])
		cv2.circle(img,(curr_col, curr_row), 1, (0,255,0), -1)
		# img[curr_row,curr_col]=[0 ,255 ,0]
		# img[curr_row,curr_col,1]=255
		# img[curr_row,curr_col,2]=255
	print('here')
	# cv2.imshow('img',img)
	cv2.imwrite('../results/chicken_broth_keypoints.jpeg',img)
	pass



if __name__ == '__main__':
	visualize_keypoints(locsDoG,image)
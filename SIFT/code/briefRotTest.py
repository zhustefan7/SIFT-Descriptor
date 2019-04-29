
import cv2 
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from BRIEF import briefLite, briefMatch


def briefRotTest():
	im = cv2.imread('../data/model_chickenbroth.jpg')
	locs0, desc0 = briefLite(im)
	imH,imW,d= im.shape
	cv2.getRotationMatrix2D
	curr_theta=0
	num_matches=[]
	for i in range(37):
		M=cv2.getRotationMatrix2D((imW/2,imH/2),curr_theta,1)
		curr_im=cv2.warpAffine(im, M, im.shape[1::-1], flags=cv2.INTER_LINEAR)
		curr_locs,curr_desc=briefLite(curr_im)
		matches = briefMatch(desc0, curr_desc)
		num_matches.append(matches.shape[0])
		curr_theta=curr_theta-10


	print(len(num_matches))

	# objects = ('10 ', '', '30 deg', '40 deg', '50 deg', '60 deg','70 deg', '80 deg','90 deg','100 deg',
	# 	'110 deg', '120 deg', '130 deg', '140 deg','150 deg', '160 deg', '170 deg', '180 deg'
	# 	,'190 deg',)
	objects = ('0deg', ' ', '', '30deg', ' ', ' ', ' 60deg',' ', ' ','90deg',' ',' ', '120deg', ' ', ' ','150deg', ' ', '', '180deg'
		,' ',' ','210deg',' ',' ','240deg ',' ',' ','270deg',' ',' ','300deg','',' ','330deg',' ',' ','360deg')
	print(len(objects))
	y_pos = np.arange(len(objects))
	 
	plt.bar(y_pos, num_matches, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('Number of Matches')
	plt.title('Number of Matches Vs Degree of Rotation')
	 
	plt.show()

		# cv2.imwrite('curr_im.jpg',curr_im)










if __name__ == '__main__':
	briefRotTest()



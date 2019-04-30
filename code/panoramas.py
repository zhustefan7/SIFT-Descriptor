import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches
import math

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    H2to1=np.reshape(H2to1,(3,3))
    warped_im2=cv2.warpPerspective(im2, H2to1,(im2.shape[1]+im1.shape[1],im2.shape[0]))
    
    warped_im2[0:im1.shape[0],0:im1.shape[1],:]=im1
    pano_im=warped_im2

    return pano_im  


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...

    H2to1=np.reshape(H2to1,(3,3))
    # print(im2.shape)
    imH_2,imW_2,D=im2.shape
    imH_1,imW_1,D=im1.shape

    im2_corners=np.array([[0, imW_2, 0, imW_2],[0,0,imH_2,imH_2],[1,1,1,1]])
    warped_im2_corners=np.matmul(H2to1,im2_corners)

    corners_w=np.divide(warped_im2_corners[0,:],warped_im2_corners[2,:])
    corners_h=np.divide(warped_im2_corners[1,:],warped_im2_corners[2,:])

    min_height=np.min(corners_h)

    max_height=np.max(corners_h)
  
    max_width=np.max(corners_w)
    out_width=1500
    out_height=int(out_width*(max_height-min_height)/max_width)
    scale=out_width/max_width 
    Tx=0
    Ty=abs(min_height)
    M=np.array([[scale,0,Tx],[0,scale, Ty*scale],[0,0,1]])


    warped_im1 = cv2.warpPerspective(im1, M,(int(imW_1*scale),int(imH_1*scale)+math.floor(Ty)))
    warped_im2 = cv2.warpPerspective(im2, np.matmul(M,H2to1),(out_width,out_height))

    warped_im2[math.floor(Ty):warped_im1.shape[0],0:warped_im1.shape[1],:]=warped_im1[math.floor(Ty):warped_im1.shape[0],0:warped_im1.shape[1],:]

    return warped_im2



def generatePanorama(im1,im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2,H2to1)
    return pano_im



if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')

    im2 = cv2.imread('../data/incline_R.png')

#     locs1, desc1 = briefLite(im1)
#     locs2, desc2 = briefLite(im2)
#     # # np.save('locs1',locs1)
#     # # np.save('locs2',locs2)
    locs1=np.load('locs1.npy')
    locs2=np.load('locs2.npy')
#     # matches = briefMatch(desc1, desc2)
#     # # np.save('incline_matches',matches)
#     # # # plotMatches(im1,im2,matches,locs1,locs2)
    matches=np.load('incline_matches.npy')
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # np.save('6_1.npy',H2to1)
    H2to1=np.load('6_1.npy')
    pano_im=imageStitching_noClip(im1, im2, H2to1)
    # pano_im = generatePanorama(im1,im2)
    cv2.imwrite('../writeup_images/q6_2.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
import numpy as np
import cv2
import math

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]
    imH,imW,level_num=gaussian_pyramid.shape

    DoG_pyramid=np.empty([imH,imW,level_num-1])
    for i in range(0,level_num-1):
        DoG_pyramid[:,:,i]=gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    ##################
    # TO DO ...
    # Compute principal curvature here
    imH,imW,level_num=DoG_pyramid.shape
    principal_curvature = np.empty([imH,imW,level_num])
    for i in range(level_num):
        mask=np.ones((imH,imW))
        img=DoG_pyramid[:,:,i]
        Dx=cv2.Sobel(img,-1,1,0)
        Dy=cv2.Sobel(img,-1,0,1)
        Dxx=cv2.Sobel(Dx,-1,1,0)
        Dxy=cv2.Sobel(Dx,-1,0,1)
        Dyy=cv2.Sobel(Dy,-1,0,1)
        Dyx=cv2.Sobel(Dy,-1,1,0) 
        trace_matrix=Dxx+Dyy
        det_matrix=np.multiply(Dxx,Dyy)-np.square(Dxy)
        # print(det_matrix==0)

        mask[det_matrix==0]=0
        det_matrix[det_matrix==0]=1
        principal_curvature[:,:,i]=np.divide(np.square(trace_matrix),det_matrix)
        principal_curvature[:,:,i]=np.multiply(principal_curvature[:,:,i],mask)

    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = np.empty((0,3))
    ##############
    #  TO DO ...
    # Compute locsDoG here
    imH,imW,level_num=DoG_pyramid.shape
    x_final=np.empty((0))
    y_final=np.empty((0))
    levels_final=np.empty((0))

    for i in range(level_num):
        indices0=[]
        #finding the extrema indices
        for j in range(1,imH-1):
            for k in range(1,imW-1):
                temp=DoG_pyramid[j-1:j+2,k-1:k+2,i]
                if i==0:
                    temp=np.append(temp,DoG_pyramid[j,k,i+1])
                elif i==level_num-1:
                    temp=np.append(temp,DoG_pyramid[j,k,i-1])
                else:
                    temp=np.append(temp,[DoG_pyramid[j,k,i-1],DoG_pyramid[j,k,i+1]])

                if DoG_pyramid[j,k,i]==np.amax(temp) or DoG_pyramid[j,k,i]==np.amin(temp):
                    indices0.append(j*imW+k)

        indices0=np.array(indices0)
        pyramid_temp=np.reshape(DoG_pyramid[:,:,i],(imH*imW,1))
        curvature_temp=np.reshape(principal_curvature[:,:,i],(imH*imW,1))
        #thresholding using th_contrast
        indices1=np.where(pyramid_temp>th_contrast)[0]
        #thresholding using th_r
        indices2=np.where(curvature_temp<th_r)[0]
        #find the indices that satisfy both thresholding
        final_indices=np.intersect1d(indices1,indices2)
        final_indices=np.intersect1d(final_indices,indices0)
        # print('final_indices',final_indices.shape)
        length=len(final_indices)
        #the third column of the locsDoG
        levels=np.full((length),DoG_levels[i])
        y_indices=np.floor(np.divide(final_indices,imW))   #row number
        x_indices=np.remainder(final_indices,imW)          #column number
        x_final=np.concatenate((x_final,x_indices))
        y_final=np.concatenate((y_final,y_indices))
        levels_final=np.concatenate((levels_final,levels))
        

    locsDoG=np.stack((x_final,y_final,levels_final),axis=1)


    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here

    gauss_pyramid=createGaussianPyramid(im, sigma0, 
        k, levels)
    DoG_pyramid, DoG_levels=createDoGPyramid(gauss_pyramid,levels)
    pc_curvature=computePrincipalCurvature(DoG_pyramid)

    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, pc_curvature, th_contrast, th_r)

    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
#     displayPyramid(im_pyr)
#     # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
#     displayPyramid(DoG_pyr)
#     # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
#     # displayPyramid(pc_curvature)
#     # test get local extrema
#     th_contrast = 0.03
#     th_r = 12
#     locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
#     # test DoG detector
#     # locsDoG, gaussian_pyramid = DoGdetector(im)



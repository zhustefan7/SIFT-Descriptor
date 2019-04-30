import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
from scipy.spatial import distance
from scipy.spatial.distance import cdist

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    width=p1.shape[1]
    A=[]
    for i in range(width):
        x1=p1[0,i]
        y1=p1[1,i]
        x2=p2[0,i]
        y2=p2[1,i]
        # temp_row1=np.array([[x2,y2,1,0,0,0,-x1*x2,-x1*y2,x1]])
        # temp_row2=np.array([[0,0,0,x2,y2,1,-y1*x2,y1*y2,-y1]])
        temp_row1=np.array([[x2,y2,1,0,0,0,-x1*x2,-x1*y2,-x1]])
        temp_row2=np.array([[0,0,0,x2,y2,1,-y1*x2,-y1*y2,-y1]])

        if i==0:
            A=np.concatenate((temp_row1,temp_row2),axis=0)
        else:
            A=np.concatenate((A,temp_row1),axis=0)
            A=np.concatenate((A,temp_row2),axis=0)

    AtA=np.matmul(np.transpose(A),A)
    # print(AtA.shape)
    # U,S,V=np.linalg.svd(A, full_matrices=True, compute_uv=True)

    w,v=np.linalg.eigh(AtA)
    H2to1=v[:,0]
    # print(w)
    # print( H2to1)
    return H2to1

        
def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    bestH=None
    best_len=0
    for i in range(num_iter):
        random_indices=np.random.permutation(matches.shape[0])
        #indices of points being used to calculate the current homography model
        sample_indices=random_indices[0:4]
        #indices of all the other data points
        remain_indices=np.setdiff1d(random_indices, sample_indices)

        #sample points being used to calculate the  current homography
        sampled_point1=locs1[matches[sample_indices,:][:,0],0:2]
        sampled_point2=locs2[matches[sample_indices,:][:,1],0:2]


        H_model=computeH(np.transpose(sampled_point1),np.transpose(sampled_point2))
        H_model=np.reshape(H_model,(3,3))

        #extracting point1 and point2 vectors
        point1=locs1[matches[remain_indices,:][:,0],0:2]
        point2=locs2[matches[remain_indices,:][:,1],0:2]
        #adding in the sampled points 
        point1=np.transpose(np.concatenate((point1,sampled_point1),axis=0))
        point2=np.transpose(np.concatenate((point2,sampled_point2),axis=0))
        ones=np.ones((1,point1.shape[1]))

        #converting cartesian coordinates to homogeneous coordinates
        point1=np.concatenate((point1,ones),axis=0)
        point2=np.concatenate((point2,ones),axis=0)

        #applying the homography to all point2s
        point2_transformed=np.matmul(H_model,point2)
        point2_transformed=np.divide(point2_transformed,point2_transformed[2,:])
        #calculate the distance between transformed point2 and point1

        distance=abs(point1-point2_transformed)
        distance=np.linalg.norm(distance,axis=0)
        #applying threshol
        comparison=np.array([distance<tol]).astype(int)
        qualified_indices=np.where(comparison == 1)[1]

        if len(qualified_indices)>best_len:
            best_len=len(qualified_indices)
            bestH=H_model


    return bestH


    

# if __name__ == '__main__':
#     im1 = cv2.imread('../data/model_chickenbroth.jpg')
#     im2 = cv2.imread('../data/chickenbroth_01.jpg')
#     locs1, desc1 = briefLite(im1)
#     locs2, desc2 = briefLite(im2)
#     matches = briefMatch(desc1, desc2)
#     ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

    # p1=np.array([[1,2,3,4,5,66,7,3,4,3,45,6,2,1],[4,56,23,5,67,23,3,4,5,45,12,25,8,34]])
    # p2=np.array([[3,4,3,2,56,3,5,6,3,2,3,56,2,3],[23,23,5,6,3,23,5,67,3,3,2,45,6,2]])
    # computeH(p1,p1)

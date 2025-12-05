# Brian Bartley
# Lizza Novikova

from scipy.ndimage import map_coordinates, sobel
import numpy as np

'''
Inverse Compositional Affine Alignment implementation. Takes previous image, 
current image, and the bounding box defined by top left and bottom right 
coordinates. Returns the new bounding box after affine alignment.
'''

def InverseCompositionAffine(prevImage, currImage, boundingBox):
    # converting to grayscale
    if prevImage.ndim == 3: prevImage = np.mean(prevImage,axis=2)
    if currImage.ndim == 3: currImage = np.mean(currImage,axis=2)
    # setting up the threshold
    x1,y1,x2,y2 = boundingBox
    threshold = 0.01875
    iterations = 50
    p = np.zeros((6,1))
    # setting up the grid
    x = np.arange(x1,x2+1)
    y = np.arange(y1,y2+1)
    X, Y = np.meshgrid(x,y)
    Xf = X.ravel()
    Yf = Y.ravel()
    # compute gradients once
    Ix_full = sobel(prevImage, axis=1)
    Iy_full = sobel(prevImage, axis=0)
    # sample template and gradients
    T = map_coordinates(prevImage, [Yf, Xf], order=1)
    Ix = map_coordinates(Ix_full, [Yf, Xf], order=1)
    Iy = map_coordinates(Iy_full, [Yf, Xf], order=1)
    # evaluating the jacobian and hessian (precomputed)
    A = np.vstack((Ix*Xf, Ix*Yf, Ix, Iy*Xf, Iy*Yf, Iy)).T
    H = A.T@A
    H_inv = np.linalg.inv(H)
    # refinement loop
    for _ in range(iterations):
        # warping the image
        M = np.array([[1.0+p[0,0],p[1,0],p[2,0]],[p[3,0],1.0+p[4,0],p[5,0]]])
        wX = M[0,0]*Xf+M[0,1]*Yf+M[0,2]
        wY = M[1,0]*Xf+M[1,1]*Yf+M[1,2]
        warped = map_coordinates(currImage,[wY,wX],order=1)
        # compute the error image
        e = T-warped
        # computing the approximate minimizer
        dp = H_inv @ (A.T@e)
        # inverse compositional update
        dM = np.array([[1.0+dp[0], dp[1], dp[2]],[dp[3], 1.0+dp[4], dp[5]]]).reshape(2,3)
        dMi = np.linalg.inv(np.vstack([dM,[0,0,1]]))[:2,:]-np.array([[1,0,0],[0,1,0]])
        dpi = dMi.flatten()[:, None]
        p -= dpi
        # checking the threshold
        if np.linalg.norm(dp) < threshold:
            break
    # computing the new bounding box
    M = np.array([[1.0+p[0], p[1], p[2]],[p[3], 1.0+p[4], p[5]]]).reshape(2,3)
    newX1 = M[0,0]*x1+M[0,1]*y1+M[0,2]
    newY1 = M[1,0]*x1+M[1,1]*y1+M[1,2]
    newX2 = M[0,0]*x2+M[0,1]*y2+M[0,2]
    newY2 = M[1,0]*x2+M[1,1]*y2+M[1,2]
    newBoundingBox = (newX1,newY1,newX2,newY2)
    return newBoundingBox

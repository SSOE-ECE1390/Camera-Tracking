# Brian Bartley
# Lizza Novikova

from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import numpy as np

'''
Lucas Kanade Tracker implementation. Takes previous image, current image and the bounding box 
defined by top left and bottom right coordinates. Returns the new bounding box.
'''

def LucasKanadeTracker(prevImage,currImage,boundingBox):
    # setting up the threshold
    x1,y1,x2,y2 = boundingBox
    threshold = 0.01875
    iterations = 100
    points = np.zeros(2)
    # setting up the grid
    x = np.arange(x1,x2+1)
    y = np.arange(y1,y2+1)
    X,Y = np.meshgrid(x,y)
    # interpolating images
    prevImage = RectBivariateSpline(np.arange(prevImage.shape[0]),np.arange(prevImage.shape[1]),prevImage)
    currImage = RectBivariateSpline(np.arange(currImage.shape[0]),np.arange(currImage.shape[1]),currImage)
    # computing the gradient
    Ix = prevImage.ev(Y,X,dy=1)
    Iy = prevImage.ev(Y,X,dx=1)
    # refinement loop 
    for _ in range(iterations):
        # warping the image
        warpedImage = currImage.ev(Y+points[1],X+points[0])
        # compute the error image
        e = prevImage.ev(Y,X)-warpedImage
        # evaluating the jacobian
        A = np.vstack((Ix.ravel(),Iy.ravel())).T
        b = e.ravel()
        # computing the approximate minimizer
        dpoints,_,_,_ = np.linalg.lstsq(A,b,rcond=None)
        # updating the points
        points+=dpoints
        # checking the threshold
        if np.linalg.norm(dpoints)<threshold:
            break
    newBoundingBox = (x1+points[0],y1+points[1],x2+points[0],y2+points[1])
    return newBoundingBox


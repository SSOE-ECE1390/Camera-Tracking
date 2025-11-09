# Brian Bartley
# Lizza Novikova

from scipy.interpolate import RectBivariateSpline
import numpy as np

'''
Inverse Compositional Affine Alignment implementation. Takes previous image, 
current image, and the bounding box defined by top left and bottom right 
coordinates. Returns the new bounding box after affine alignment.
'''

def InverseCompositionAffine(prevImage, currImage, boundingBox):
    # converting to grayscale
    prevImage = np.mean(prevImage, axis=2)
    currImage = np.mean(currImage, axis=2)
    # setting up the threshold
    x1, y1, x2, y2 = boundingBox
    threshold = 0.01875
    iterations = 100
    p = np.zeros((6, 1))
    # setting up the grid
    x = np.arange(x1, x2 + 1)
    y = np.arange(y1, y2 + 1)
    X, Y = np.meshgrid(x, y)
    # interpolating images
    prevImage = RectBivariateSpline(np.arange(prevImage.shape[0]), np.arange(prevImage.shape[1]), prevImage)
    currImage = RectBivariateSpline(np.arange(currImage.shape[0]), np.arange(currImage.shape[1]), currImage)
    # computing the gradient
    Ix = prevImage.ev(Y, X, dy=1)
    Iy = prevImage.ev(Y, X, dx=1)
    # evaluating the jacobian and hessian
    A = np.vstack((Ix * X, Ix * Y, Ix, Iy * X, Iy * Y, Iy)).T
    H = A.T @ A
    # refinement loop
    for _ in range(iterations):
        # warping the image
        M = np.array([[1.0 + p[0, 0], p[1, 0], p[2, 0]],
                      [p[3, 0], 1.0 + p[4, 0], p[5, 0]]])
        wX = M[0, 0] * X + M[0, 1] * Y + M[0, 2]
        wY = M[1, 0] * X + M[1, 1] * Y + M[1, 2]
        warpedImage = currImage.ev(wY, wX)
        # compute the error image
        e = prevImage.ev(Y, X) - warpedImage
        # computing the approximate minimizer
        b = e.ravel()
        dp, _, _, _ = np.linalg.lstsq(H, A.T @ b, rcond=None)
        # inverse compositional update
        dM = np.array([[1.0 + dp[0], dp[1], dp[2]],
                       [dp[3], 1.0 + dp[4], dp[5]]]).reshape(2, 3)
        dMi = np.linalg.inv(np.vstack([dM, [0, 0, 1]])) - np.eye(3)
        dpi = dMi[:2].flatten()
        p -= dpi[:, None]
        # checking the threshold
        if np.linalg.norm(dp) < threshold:
            break
    # computing the new bounding box
    M = np.array([[1.0 + p[0], p[1], p[2]],
                  [p[3], 1.0 + p[4], p[5]]]).reshape(2, 3)
    newX1 = M[0, 0] * x1 + M[0, 1] * y1 + M[0, 2]
    newY1 = M[1, 0] * x1 + M[1, 1] * y1 + M[1, 2]
    newX2 = M[0, 0] * x2 + M[0, 1] * y2 + M[0, 2]
    newY2 = M[1, 0] * x2 + M[1, 1] * y2 + M[1, 2]
    newBoundingBox = (newX1, newY1, newX2, newY2)
    return newBoundingBox

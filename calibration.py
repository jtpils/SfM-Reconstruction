# taken from OpenCV tutorials

import cv2
import numpy as np
import glob

nx = 9
ny = 7

# image resolution
img_h = 2592
img_w = 4608

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((nx*ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)*30

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# create a folder called calibration and store chessboard images in there
images = glob.glob('calibration/*.jpg')

i = 0

for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points (after refining them)
    if ret:

        i += 1
        print(i)
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img_h, img_w), None, None)
np.save('OP5_calibration_matrix', mtx)

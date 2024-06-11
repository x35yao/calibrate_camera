import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R

date = '2024-01-02'

frameSize = (2208, 1242)
cameraMatrixL = np.array([[1068.28, 0, 1138.21],
                         [0, 1068.73, 658.0130],
                         [0, 0, 1]])

cameraMatrixR = np.array([[1062.54, 0, 1143.5699],
                         [0, 1062.89, 648.2990],
                         [0, 0, 1]])
k1L, k2L, p1L, p2L, k3L = -0.0520, 0.0256, 0.0001, -0.0005, -0.0106
k1R, k2R, p1R, p2R, k3R = -0.0507, 0.0208, 0.0001, -0.0005, -0.0106
distL = np.array([k1L, k2L, p1L, p2L, k3L])
distR = np.array([k1R, k2R, p1R, p2R, k3R])

rx, ry, rz = -0.0007, -0.0017, -0.0003
rot = R.inv(R.from_rotvec([rx, ry, rz])).as_matrix()
T = - np.array([119.87, -0.0182, -0.7044])

S = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
essentialMatrix = rot @ S
fundamentalMatrix = np.linalg.inv(cameraMatrixL.T) @ essentialMatrix @ np.linalg.inv(cameraMatrixR)

rectifyScale = 0.9
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(cameraMatrixL, distL,
                                                                           cameraMatrixR, distR,
                                                                           frameSize, rot, T,
                                                                               rectifyScale, (0, 0))


result = {}
result['left-right'] = {}
result['left-right']['cameraMatrix1'] = cameraMatrixL
result['left-right']['cameraMatrix2'] = cameraMatrixR
result['left-right']['distCoeffs1'] = distL
result['left-right']['distCoeffs2'] = distR

result['left-right']['R'] = rot
result['left-right']['T'] = T
result['left-right']['R1'] = rectL
result['left-right']['R2'] = rectR
result['left-right']['P1'] = projMatrixL
result['left-right']['P2'] = projMatrixR
result['left-right']['F'] = fundamentalMatrix
result['left-right']['E'] = essentialMatrix
result['left-right']['roi1'] = roi_L
result['left-right']['roi2'] = roi_R
result['left-right']['Q'] = Q
result['left-right']['image_shape'] = [frameSize, frameSize]

import os
import pickle
os.makedirs(f'./camera_matrix/{date}', exist_ok=True)
fname = f'./camera_matrix/{date}/stereo_params.pickle'

if os.path.isfile(fname):
    print('Removing existed file!')
    os.remove(fname)
with open(fname, 'wb') as f:
    pickle.dump(result, f)
print(result)
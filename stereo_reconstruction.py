#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import time
import struct

# https://vision.middlebury.edu/stereo/data/scenes2014/ --datasets
#https://answers.opencv.org/question/228119/disparity-map-to-point-cloud-gone-wrong/
"""
SCENE-{perfect,imperfect}/     -- each scene comes with perfect and imperfect calibration (see paper)
  ambient/                     -- directory of all input views under ambient lighting
    L{1,2,...}/                -- different lighting conditions
      im0e{0,1,2,...}.png      -- left view under different exposures
      im1e{0,1,2,...}.png      -- right view under different exposures
  calib.txt                    -- calibration information
  im{0,1}.png                  -- default left and right view
  im1E.png                     -- default right view under different exposure
  im1L.png                     -- default right view with different lighting
  disp{0,1}.pfm                -- left and right GT disparities
  disp{0,1}-n.png              -- left and right GT number of samples (* perfect only)
  disp{0,1}-sd.pfm             -- left and right GT sample standard deviations (* perfect only)
  disp{0,1}y.pfm               -- left and right GT y-disparities (* imperfect only)
"""
#calibration information
"""
cam0=[3979.911 0 1244.772; 0 3979.911 1019.507; 0 0 1]
cam1=[3979.911 0 1369.115; 0 3979.911 1019.507; 0 0 1]
doffs=124.343
baseline=193.001
width=2964
height=2000
ndisp=270
isint=0
vmin=23
vmax=245
dyavg=0
dymax=0
"""
def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
    rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    # Write header of .ply file
    print("opening file")
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
      if(i%1000 == 0):
        print(i)
      fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()

max_disparity = 245
min_disparity = 23
num_disparities = max_disparity - min_disparity
window_size = 5
stereo = cv.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities, blockSize = 5, uniquenessRatio = 5, speckleWindowSize = 5, speckleRange = 5, disp12MaxDiff = 0, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2)
imgL = cv.imread('C:\\Users\\sushl\\Desktop\\VO\\stereo-reconstruction\\ima0.png',0)
print(imgL.shape)
imgR = cv.imread('C:\\Users\\sushl\\Desktop\\VO\\stereo-reconstruction\\ima1.png',0)
print(imgR.shape)
disparity = stereo.compute(imgL,imgR).astype(np.float32)

k = np.array([[3979.911, 0 ,1244.772], 
            [0, 3979.911,1019.507], 
            [0, 0, 1]])
            
distortion = np.zeros((5,1)).astype(np.float32)
T = np.zeros((3,1))
T[1,0] = 193.001
#print(T)
R1,R2,P1,P2,Q,_,_ = cv.stereoRectify(k,distortion,k,distortion,imgL.shape,np.identity(3),T)
point_cloud = cv.reprojectImageTo3D(disparity,Q)
mask = disparity > disparity.min()
xp=point_cloud[:,:,0]
yp=point_cloud[:,:,1]
zp=point_cloud[:,:,2]
xp = xp[np.where(mask== True)[0],np.where(mask== True)[1]]
yp = yp[np.where(mask== True)[0],np.where(mask== True)[1]]
zp = zp[np.where(mask== True)[0],np.where(mask== True)[1]]

xp=xp.flatten().reshape(-1,1)
yp=yp.flatten().reshape(-1,1)
zp=zp.flatten().reshape(-1,1)
point3d = np.hstack((xp,yp,zp))
#print(np.where(mask== True)[0])
write_pointcloud("C:\\Users\\sushl\\Desktop\\VO\\stereo-reconstruction\\stereo.ply",point3d)

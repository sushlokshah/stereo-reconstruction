#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import time
import struct
import pandas as pd

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

cam0=[5806.559 0 1429.219; 0 5806.559 993.403; 0 0 1]
cam1=[5806.559 0 1543.51; 0 5806.559 993.403; 0 0 1]
doffs=114.291
baseline=174.019
width=2960
height=2016
ndisp=250
isint=0
vmin=38
vmax=222
dyavg=0
dymax=0

cam0=[5299.313 0 1263.818; 0 5299.313 977.763; 0 0 1]
cam1=[5299.313 0 1438.004; 0 5299.313 977.763; 0 0 1]
doffs=174.186
baseline=177.288
width=2988
height=2008
ndisp=180
isint=0
vmin=54
vmax=147
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
    #rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
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
      if(i%5000 == 0):
        print(i)
      fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()

def config_info(path):
  file1 = open(path, 'r')
  Lines = file1.readlines()
  Dic = {}
  for l in Lines:
    a = l.split("=")
    if(a[1][-1]== "\n"):
      try:
        Dic[a[0]] = int(a[1][:-1])
      except:
        Dic[a[0]] = a[1][:-1]
    else:
      try:
        Dic[a[0]] = int(a[1])
      except:
        Dic[a[0]] = a[1]
  Dic["cam0"] = Dic["cam0"].split(" ")
  Dic["cam0"][0] = Dic["cam0"][0][1:]
  Dic["cam0"][2] = Dic["cam0"][2][:-1]
  Dic["cam0"][5] = Dic["cam0"][5][:-1]
  Dic["cam0"][-1] = Dic["cam0"][-1][:-1]
  Dic["cam1"] = Dic["cam1"].split(" ")
  Dic["cam1"][0] = Dic["cam1"][0][1:]
  Dic["cam1"][2] = Dic["cam1"][2][:-1]
  Dic["cam1"][5] = Dic["cam1"][5][:-1]
  Dic["cam1"][-1] = Dic["cam1"][-1][:-1]
  for i in range(len(Dic["cam0"])):
    Dic["cam0"][i] = float(Dic["cam0"][i])
    Dic["cam1"][i] = float(Dic["cam1"][i])
  Dic["cam0"] = np.array(Dic["cam0"]).reshape(3,3)
  Dic["cam1"] = np.array(Dic["cam1"]).reshape(3,3)
  max_disparity = Dic["vmax"]
  min_disparity = Dic["vmin"]
  num_disparities = max_disparity - min_disparity
  window_size = 5
  k = Dic["cam0"]
  distortion = np.zeros((5,1)).astype(np.float32)
  T = np.zeros((3,1))
  T[0,0] = Dic["baseline"]
  R1,R2,P1,P2,Q,_,_ = cv.stereoRectify(k,distortion,k,distortion,(Dic["height"],Dic["width"]),np.identity(3),T)
  return Dic,k,Q,max_disparity,min_disparity,num_disparities,window_size

def find_disparity(image1,image2,path):
    Dic,k,Q,max_disparity,min_disparity,num_disparities,window_size = config_info(path)
    stereo = cv.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities, blockSize = 5, uniquenessRatio = 2, speckleWindowSize = 5, speckleRange = 5, disp12MaxDiff = 10, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2)
    imgL = cv.imread(image1,0)
    print(imgL.shape)
    imgR = cv.imread(image2,0)
    print(imgR.shape)
    disparity = stereo.compute(imgL,imgR).astype(np.float32)
    plt.imshow(disparity,"jet")
    #plt.pause(0.1)
    plt.show()
    cv.imwrite("disparity.jpg",disparity)
    return disparity,Q

def disparity_to_pointcloud(disparity,Q,image1):
  color = cv.imread(image1)
  print(color.shape)
  point_cloud = cv.reprojectImageTo3D(disparity,Q)
  mask = disparity > disparity.min()
  xp=point_cloud[:,:,0]
  yp=point_cloud[:,:,1]
  zp=point_cloud[:,:,2]
  color = color[mask]
  color = color.reshape(-1,3)
  xp = xp[np.where(mask== True)[0],np.where(mask== True)[1]]
  yp = yp[np.where(mask== True)[0],np.where(mask== True)[1]]
  zp = zp[np.where(mask== True)[0],np.where(mask== True)[1]]

  xp=xp.flatten().reshape(-1,1)
  yp=yp.flatten().reshape(-1,1)
  zp=zp.flatten().reshape(-1,1)
  point3d = np.hstack((xp,yp,zp))
  return point3d,color

if __name__ == "__main__":
    disparity,Q = find_disparity("imc0.png","imc1.png","calibration_info.txt")
    point_cloud,color = disparity_to_pointcloud(disparity,Q,"imc0.png")
    write_pointcloud("C:\\Users\\sushl\\Desktop\\VO\\stereo-reconstruction\\stereo4.ply",point_cloud,color)

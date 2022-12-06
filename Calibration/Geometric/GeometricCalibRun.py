#This file exemplifies the use of the GeometricCalib class

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from GeometricCalib import GeometricCalib
from Calibration.Geometric.util import rescale, crop, display

import pickle


### Calibration
#Images path
path = r"C:\Users\chris\Documents\Maitrise\Data\Calibration\Geomteric\Images 5D\5 mars\All\pano\\"

calib = GeometricCalib()
calib.calibrate(path, ".JPG", output="calib5D.pkl", CHECKERBOARD=(10,7))



### Apply the cosine correction to an image with the calibration file

#img_path = r"C:\Users\chris\OneDrive\Documents\Universite\Hiver 2022\Code\calibration\geomteric\CameraCervo\2022_07_07_17_22_52.jpg"
#img = cv2.imread(img_path)


#calib.loadKD("calib5D.pkl")

#res = calib.createHemisphericalSelf(img)

#display(res)




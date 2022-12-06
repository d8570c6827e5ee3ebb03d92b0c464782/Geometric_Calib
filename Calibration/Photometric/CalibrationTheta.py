from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"
import os,sys

from Geometric_Calib.Calibration.Photometric.envmap.environmentmap import EnvironmentMap
import pickle
import matplotlib.pyplot as plt

import cv2
import numpy as np
from glob import glob
import csv

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from mpl_interactions import ioff, panhandler, zoom_factory
from mplcursors import cursor


#assume que pour chaque image xyz.exr, il existe un fichier xyz.csv avec les données XYZ de du CL200a (en une ligne)

#Le path qui contient les images HDRs (une des deux caméras)
HDRs_root_path = ""
#La calibration géométrique de la caméra
geometric_calib_path = ""
#L'image de vignetting exporté par PT-gui
vignetting_mask_path = ""


HDRs = glob(HDRs_root_path + "/**/*.exr", recursive=True)

with open(geometric_calib_path, 'rb') as f:
    k, d = pickle.load(f)

vign_mask = cv2.imread(vignetting_mask_path, cv2.IMREAD_UNCHANGED)


def computeIllumiance(img_path):

    e = EnvironmentMap(img_path, 'fisheye', k=k,d=d,xi=1.0)
    e_sa = e.solidAngles()

    vectors = e.worldCoordinates()
    vectors_x = vectors[0]
    vectors_y = vectors[1]
    vectors_z = vectors[2]
    n = np.array([0,0,-1]) #Front dir -Z

    #Stack 3 direction vectors
    vectors_top = np.stack([vectors_x,vectors_y,vectors_z], axis=-1)

    #Compute cos angle between front dir and each pixel
    vectors_norm = np.linalg.norm(vectors_top, axis = 2)
    vectors_dot = np.dot(vectors_top, n)
    cos = vectors_dot/vectors_norm

    mask = np.where(cos >= 0, 1, 0)
    mask = np.where(np.isnan(e_sa), 0, 1)
    e_sa = np.nan_to_num(e_sa)

    #Compute illuminance
    illumiance_r = np.sum(e_sa * cos * e.data[:,:,0] * mask)
    illumiance_g = np.sum(e_sa * cos * e.data[:,:,1] * mask)
    illumiance_b = np.sum(e_sa * cos * e.data[:,:,2] * mask)
    #print(illumiance_r, illumiance_g, illumiance_b)

    return illumiance_r, illumiance_g, illumiance_b

def XYZ2RGB(X,Y,Z):
    R = 3.2404542*X - 0.9692660*Y + 0.0556434*Z
    G = -1.5371385*X + 1.8760108*Y - 0.2040259*Z
    B = -0.4985314*X + 0.0415560*Y + 1.0572252*Z
    return R,G,B

R_theta = np.array([])
G_theta = np.array([])
B_theta = np.array([])

R_CL200a = np.array([])
G_CL200a = np.array([])
B_CL200a = np.array([])

files = []

for HDR in HDRs:
    img = cv2.imread(HDR, cv2.IMREAD_UNCHANGED)
    #Apply vignetting correction
    img = img * vign_mask

    #compute illuminance
    illumiance = computeIllumiance(img)

    #Save illuminance
    R_theta = np.append(R_theta, illumiance[0])
    G_theta = np.append(G_theta, illumiance[1])
    B_theta = np.append(B_theta, illumiance[2])

    #Read CL200a data
    csv_path = HDR[:-4] + ".csv"
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        row1 = next(reader)
        X = float(row1[0])
        Y = float(row1[1])
        Z = float(row1[2])
        R,G,B = XYZ2RGB(X,Y,Z)
        R_CL200a = np.append(R_CL200a, R)
        G_CL200a = np.append(G_CL200a, G)
        B_CL200a = np.append(B_CL200a, B)

    files.append(HDR)


#Compute regression lines for each channel
with ioff:
    fig, ax = plt.subplots(1, 1)

ax[0].set_title('f/14')
ax[0].scatter(R_theta, R_CL200a, color="red")
ax[0].scatter(G_theta, G_CL200a, color="green")
ax[0].scatter(B_theta, B_CL200a, color="blue")

#Regression
print('Calibration coefficients')
regr_R = linear_model.LinearRegression(fit_intercept = False)
regr_R.fit(R_theta.reshape(-1, 1), R_CL200a.reshape(-1, 1))
R_pred = regr_R.predict(R_theta.reshape(-1, 1)).reshape(-1)
ax[0].plot(R_theta, R_pred, color="r", linestyle='dotted')
print("R_coeffs: ", regr_R.coef_, " + ", regr_R.intercept_)
print("R_r2: ", r2_score(R_CL200a, R_pred))

regr_G = linear_model.LinearRegression(fit_intercept = False)
regr_G.fit(G_theta.reshape(-1, 1), G_CL200a.reshape(-1, 1))
G_pred = regr_G.predict(G_theta.reshape(-1, 1)).reshape(-1)
ax[0].plot(G_theta, G_pred, color="g", linestyle='dotted')
print("G_coeffs: ", regr_G.coef_, " + ", regr_G.intercept_)
print("G_r2: ", r2_score(G_CL200a, G_pred))

regr_B = linear_model.LinearRegression(fit_intercept = False)
regr_B.fit(B_theta.reshape(-1, 1), B_CL200a.reshape(-1, 1))
B_pred = regr_B.predict(B_theta.reshape(-1, 1)).reshape(-1)
ax[0].plot(B_theta, B_pred, color="b", linestyle='dotted')
print("B_coeffs: ", regr_B.coef_, " + ", regr_B.intercept_)
print("B_r2: ", r2_score(B_CL200a, B_pred))

fig.supxlabel('Computed per channel illuminance')
fig.supylabel('Measured per channel illuminance')
ax[0].grid()

disconnect_zoom = zoom_factory(ax[0])
pan_handler = panhandler(fig)

cursor(ax[0]).connect(
    "add", lambda sel: sel.annotation.set_text(files[sel.index]))

plt.show()

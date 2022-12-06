import numpy as np
from numpy import logical_and as land, logical_or as lor
import matplotlib.pyplot as plt


def world2latlong(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    a latitude-longitude map."""
    u = 1 + (1 / np.pi) * np.arctan2(x, -z)
    v = (1 / np.pi) * np.arccos(y)
    # because we want [0,1] interval
    u = u / 2
    return u, v


def world2skylatlong(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    a sky-latitude-longitude map (the zenith hemisphere of a latlong map)."""
    u = 1 + (1 / np.pi) * np.arctan2(x, -z)
    v = (1 / np.pi) * np.arccos(y) * 2
    # because we want [0,1] interval
    u = u / 2
    return u, v


def world2angular(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    an angular map."""
    # world -> angular

    # take advantage of the division by zero handling of numpy
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    denum = (2 * np.pi * np.sqrt(x**2 + y**2))
    rAngular = np.arccos(-z) / denum
    v = np.atleast_1d(0.5 - rAngular * y)
    u = np.atleast_1d(0.5 + rAngular * x)
    
    u[~np.isfinite(rAngular)] = 0.5
    # handle [0, 0, -1]
    v[np.isnan(rAngular)] = 0.5
    # handle [0, 0, 1]
    v[np.isinf(rAngular)] = 0.

    if u.size == 1:
        return u.item(), v.item()

    return u, v


def latlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def skylatlong2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a latlong map."""
    u = u * 2

    # lat-long -> world
    thetaLatLong = np.pi * (u - 1)
    phiLatLong = np.pi * v / 2

    x = np.sin(phiLatLong) * np.sin(thetaLatLong)
    y = np.cos(phiLatLong)
    z = -np.sin(phiLatLong) * np.cos(thetaLatLong)

    valid = np.ones(x.shape, dtype='bool')
    return x, y, z, valid


def angular2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for an angular map."""
    # angular -> world
    thetaAngular = np.arctan2(-2 * v + 1, 2 * u - 1)
    phiAngular = np.pi * np.sqrt((2 * u - 1)**2 + (2 * v - 1)**2)

    x = np.sin(phiAngular) * np.cos(thetaAngular)
    y = np.sin(phiAngular) * np.sin(thetaAngular)
    z = -np.cos(phiAngular)

    r = (u - 0.5)**2 + (v - 0.5)**2
    valid = r <= .25  # .5**2

    return x, y, z, valid


def skyangular2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a sky angular map."""
    # skyangular -> world
    thetaAngular = np.arctan2(-2 * v + 1, 2 * u - 1)  # azimuth
    phiAngular = np.pi / 2 * np.sqrt((2 * u - 1)**2 + (2 * v - 1)**2)  # zenith

    x = np.sin(phiAngular) * np.cos(thetaAngular)
    z = np.sin(phiAngular) * np.sin(thetaAngular)
    y = np.cos(phiAngular)

    r = (u - 0.5)**2 + (v - 0.5)**2
    valid = r <= .25  # .5^2

    return x, y, z, valid

def fisheye2world(u, v, k, d, xi):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for a fisheye map."""
    #Based on opencv omnidir implementation:
    #https://github.com/opencv/opencv_contrib/blob/4.x/modules/ccalib/src/omnidir.cpp

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]
    s  = k[0, 1]
    
    k1 = d[0, 0]
    k2 = d[0, 1]
    p1 = d[0, 2]
    p2 = d[0, 3]

    u = u*u.shape[0]
    v = v*v.shape[1]

    initial_size = u.shape

    points = np.stack((u,v),axis=-1).reshape((initial_size[0]*initial_size[1],2))
    n = points.shape[0]

    #Convert to plane
    points_p = np.zeros((n,2))
    points_p[:,0] = (points[:,0]*fy-cx*fy-s*(points[:,1]-cy))/(fx*fy)
    points_p[:,1] = (points[:,1]-cy)/fy
    #np.array([(points[:,0]*fy-cx*fy-s*(points[:,1]-cy))/(fx*fy), (points[:,1]-cy)/fy])

    #Remove distortion iteratively
    points_u = np.zeros((n,2))

    for i in range(20):
        r2 = points_u[:,0]**2 + points_u[:,1]**2
        r4 = r2**2

        points_u[:,0] = (points_p[:,0] - 2*p1*points_u[:,0]*points_u[:,1] - p2*(r2+2*points_u[:,0]*points_u[:,0])) / (1 + k1*r2 + k2*r4)
        points_u[:,1] = (points_p[:,1] - 2*p2*points_u[:,0]*points_u[:,1] - p1*(r2+2*points_u[:,1]*points_u[:,1])) / (1 + k1*r2 + k2*r4)

    #Project to unit sphere
    r2 = points_u[:,0]**2 + points_u[:,1]**2
    a = (r2 + 1)
    b = 2*xi*r2
    cc = r2*xi*xi-1
    Zs = (-b + np.sqrt(b*b - 4*a*cc))/(2*a)
    
    #Y and Z negative to fit with skylibs coordinates
    x = points_u[:,0]*(Zs+xi)
    y = -points_u[:,1]*(Zs+xi)
    z = -Zs

    x = x.reshape(initial_size)
    y = y.reshape(initial_size)
    z = z.reshape(initial_size)

    r = np.sqrt((u - cx)**2 + (v - cy)**2)

    valid = r <= np.min([cx,cy, initial_size[0]-cx, initial_size[1]-cy])

    return x, y, z, valid


def world2skyangular(x, y, z):
    """Get the (u, v) coordinates of the point defined by (x, y, z) for
    a sky angular map."""
    # world -> skyangular
    thetaAngular = np.arctan2(x, z)  # azimuth
    phiAngular = np.arctan2(np.sqrt(x**2 + z**2), y)  # zenith

    r = phiAngular / (np.pi / 2)

    u = 1. / 2 + r * np.sin(thetaAngular) / 2 
    v = 1. / 2 - r * np.cos(thetaAngular) / 2

    return u, v


def sphere2world(u, v):
    """Get the (x, y, z, valid) coordinates of the point defined by (u, v)
    for the sphere map."""
    u = u * 2 - 1
    v = v * 2 - 1

    # sphere -> world
    r = np.sqrt(u**2 + v**2)
    theta = np.arctan2(u, -v)

    phi = np.zeros(theta.shape)
    valid = r <= 1
    phi[valid] = 2 * np.arcsin(r[valid])

    x = np.sin(phi) * np.sin(theta)
    y = np.sin(phi) * np.cos(theta)
    z = -np.cos(phi)
    return x, y, z, valid


def world2sphere(x, y, z):
    # world -> sphere

    # take advantage of the division by zero handling of numpy
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    
    denum = (2 * np.sqrt(x**2 + y**2))
    r = np.sin(.5 * np.arccos(-z)) / denum

    u = np.atleast_1d(.5 + r * x)
    v = np.atleast_1d(.5 - r * y)

    u[~np.isfinite(r)] = 0.5
    # handle [0, 0, -1]
    v[np.isnan(r)] = 0.5
    # handle [0, 0, 1]
    v[np.isinf(r)] = 0.

    if u.size == 1:
        return u.item(), v.item()

    return u, v


def world2cube(x, y, z):
    # world -> cube
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))
    z = np.atleast_1d(np.asarray(z))
    u = np.zeros(x.shape)
    v = np.zeros(x.shape)

    # forward
    indForward = np.nonzero(
        land(land(z <= 0, z <= -np.abs(x)), z <= -np.abs(y)))
    u[indForward] = 1.5 - 0.5 * x[indForward] / z[indForward]
    v[indForward] = 1.5 + 0.5 * y[indForward] / z[indForward]

    # backward
    indBackward = np.nonzero(
        land(land(z >= 0,  z >= np.abs(x)),  z >= np.abs(y)))
    u[indBackward] = 1.5 + 0.5 * x[indBackward] / z[indBackward]
    v[indBackward] = 3.5 + 0.5 * y[indBackward] / z[indBackward]

    # down
    indDown = np.nonzero(
        land(land(y <= 0,  y <= -np.abs(x)),  y <= -np.abs(z)))
    u[indDown] = 1.5 - 0.5 * x[indDown] / y[indDown]
    v[indDown] = 2.5 - 0.5 * z[indDown] / y[indDown]

    # up
    indUp = np.nonzero(land(land(y >= 0,  y >= np.abs(x)),  y >= np.abs(z)))
    u[indUp] = 1.5 + 0.5 * x[indUp] / y[indUp]
    v[indUp] = 0.5 - 0.5 * z[indUp] / y[indUp]

    # left
    indLeft = np.nonzero(
        land(land(x <= 0,  x <= -np.abs(y)),  x <= -np.abs(z)))
    u[indLeft] = 0.5 + 0.5 * z[indLeft] / x[indLeft]
    v[indLeft] = 1.5 + 0.5 * y[indLeft] / x[indLeft]

    # right
    indRight = np.nonzero(land(land(x >= 0,  x >= np.abs(y)),  x >= np.abs(z)))
    u[indRight] = 2.5 + 0.5 * z[indRight] / x[indRight]
    v[indRight] = 1.5 - 0.5 * y[indRight] / x[indRight]

    # bring back in the [0,1] intervals
    u = u / 3.
    v = v / 4.
    return u, v


def cube2world(u, v):
    # [u,v] = meshgrid(0:3/(3*dim-1):3, 0:4/(4*dim-1):4);
    # u and v are in the [0,1] interval, so put them back to [0,3]
    # and [0,4]
    u = u * 3
    v = v * 4

    x = np.zeros(u.shape)
    y = np.zeros(u.shape)
    z = np.zeros(u.shape)
    valid = np.zeros(u.shape, dtype='bool')

    # up
    indUp = land(land(u >= 1, u < 2), v < 1)
    x[indUp] = (u[indUp] - 1.5) * 2
    y[indUp] = 1
    z[indUp] = (v[indUp] - 0.5) * -2

    # left
    indLeft = land(land(u < 1, v >= 1), v < 2)
    x[indLeft] = -1
    y[indLeft] = (v[indLeft] - 1.5) * -2
    z[indLeft] = (u[indLeft] - 0.5) * -2

    # forward
    indForward = land(land(land(u >= 1, u < 2), v >= 1), v < 2)
    x[indForward] = (u[indForward] - 1.5) * 2
    y[indForward] = (v[indForward] - 1.5) * -2
    z[indForward] = -1

    # right
    indRight = land(land(u >= 2, v >= 1), v < 2)
    x[indRight] = 1
    y[indRight] = (v[indRight] - 1.5) * -2
    z[indRight] = (u[indRight] - 2.5) * 2

    # down
    indDown = land(land(land(u >= 1, u < 2), v >= 2), v < 3)
    x[indDown] = (u[indDown] - 1.5) * 2
    y[indDown] = -1
    z[indDown] = (v[indDown] - 2.5) * 2

    # backward
    indBackward = land(land(u >= 1, u < 2), v >= 3)
    x[indBackward] = (u[indBackward] - 1.5) * 2
    y[indBackward] = (v[indBackward] - 3.5) * 2
    z[indBackward] = 1

    # normalize
    # np.hypot(x, y, z) #sqrt(x.^2 + y.^2 + z.^2);
    norm = np.sqrt(x**2 + y**2 + z**2)
    x = x / norm
    y = y / norm
    z = z / norm

    # return valid indices
    valid_ind = lor(
        lor(lor(indUp, indLeft), lor(indForward, indRight)), lor(indDown, indBackward))
    valid[valid_ind] = 1
    return x, y, z, valid

def world2fisheye(x, y, z, k, d, xi):
    # world -> fisheye

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]
    s  = k[0, 1]
    
    k1 = d[0, 0]
    k2 = d[0, 1]
    p1 = d[0, 2]
    p2 = d[0, 3]

    initial_size = x.shape

    points = np.stack((x,y,z), axis=-1).reshape((initial_size[0]*initial_size[1], 3))
    n = points.shape[0]

    #Y and Z negative to fit with skylibs coordinates
    points[:,1] = -points[:,1]
    points[:,2] = -points[:,2]

    #Convert to unit sphere
    points_s = points / np.linalg.norm(points,axis=1)[:, np.newaxis]
    
    #Convert to normalised image plane
    points_u = np.zeros((n,2))
    points_u[:,0] = points_s[:,0]/(points_s[:,2]+xi)
    points_u[:,1] = points_s[:,1]/(points_s[:,2]+xi)
    #np.array([points_s[:,0]/(points_s[:,2]+xi), points_s[:,1]/(points_s[:,2]+xi)])

    #Add distortion
    r2 = points_u[:,0]**2 + points_u[:,1]**2
    r4 = r2**2

    points_d = np.zeros((n,2))
    points_d[:,0] = points_u[:,0] * (1 + k1*r2 + k2*r4) + 2*p1*points_u[:,0]*points_u[:,1] + p2*(r2 + 2*(points_u[:,0]**2))
    points_d[:,1] = points_u[:,1] * (1 + k1*r2 + k2*r4) + 2*p2*points_u[:,0]*points_u[:,1] + p1*(r2 + 2*(points_u[:,1]**2))

    #Convert to pixel coordinates
    u = fx*points_d[:,0] + s*points_d[:,1] + cx
    v = fy*points_d[:,1] + cy

    u = u.reshape(initial_size)
    v = v.reshape(initial_size)

    #Normalisation assumes that the image is square and that the whole image is in the input points
    u = u / initial_size[0]
    v = v / initial_size[1]

    return u, v
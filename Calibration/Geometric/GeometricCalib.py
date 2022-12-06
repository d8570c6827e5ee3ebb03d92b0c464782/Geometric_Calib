import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

#Open3D was used to visualize the point cloud of projected points
#import open3d as o3d

from Geometric_Calib.Calibration.Geometric.util import rescale, crop, display

# Class that treats the geometric calibration of fisheye lens
class GeometricCalib:

    def __init__(self):
        self.K = None
        self.D = None

    def loadKD(self, path):
        with open(path, 'rb') as f:
            K, D = pickle.load(f)
            self.K = K
            self.D = D

    def createMask(self, img):
        self.center = np.array([self.K[0,2],self.K[1,2]])
        Px = 0.
        Py = -1.
        Pz = np.cos(np.arcsin(1))
        P = np.array([[[Px,Py,Pz]]])

        rvec = np.array([0., 0., 0.])
        tvec = np.array([0., 0., 0.])
        ImPoints, _ = cv2.omnidir.projectPoints(P,rvec, tvec, self.K,1, self.D)
        
        r = abs(ImPoints[0,0,1] - self.center[1])
        Cx = int(self.center[0])
        Cy = int(self.center[1])
        
        self.mask = np.zeros((img.shape[0],img.shape[1]))
        self.mask = cv2.circle(self.mask, (Cx,Cy), int(r), (255), -1)
        

    def calibrate(self,imgs_path, img_ext, output=None, CHECKERBOARD=(10,7)):
        # Checkboard dimensions
        
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)


        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        _img_shape = None
        images = glob.glob(imgs_path+"*"+img_ext)
        count = 0
        for fname in images:
            count +=1
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            print("image:",count)
            ret, corners = self.findCheckboard(gray, CHECKERBOARD)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)

        # calculate K & DS
        K = np.zeros((3, 3), dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)
        dims = gray.shape[::-1]
        rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for _ in range(len(objpoints))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float32)  for _ in range(len(objpoints))]

        rms, K, xi, D, rvecs, tvecs, idx  =  cv2.omnidir.calibrate(
                    objectPoints=objpoints, 
                    imagePoints=imgpoints, 
                    size=dims, 
                    K=None, xi=None, D=None,
                    flags=cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER,
                    criteria=subpix_criteria)

        print("K:",K)
        print("D:",D)

        self.K = K
        self.D = D

        if output is not None:
            with open(output, 'wb') as f:
                pickle.dump([K,D], f)

    def findCheckboard(self, img, CHECKERBOARD):
        
        #small
        small_img, scale = rescale(img,1500)
        ret, corners = cv2.findChessboardCorners(small_img, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if not ret:
            print("Error")
            return False, None
        print("succes")
        
        maxi = (np.max(corners, axis=0)[0] / scale).astype(int)
        mini = (np.min(corners, axis=0)[0] / scale).astype(int)
        
        cropped, offset = crop(img, mini, maxi, 150)
        
        ret, corners = cv2.findChessboardCorners(cropped, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            corners = (corners + offset).astype(np.float32)
            #drawn_frame = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            #imshow(rescale(drawn_frame,1000)[0])
            
        return ret, corners
    
    def createEquiSelf(self, img, rvec=None):
        return self.createEqui(self.K, self.D, img, rvec=rvec)

    #http://paulbourke.net/dome/dualfish2sphere/diagram.pdf
    def createEqui(self, K,D, img, res_size=None, rvec=None):
        if res_size == None:
            #res_size = (img.shape[1],img.shape[0])
            res_size = (6570,3285)
        #res_size = (1920,1080)
        
        longs = np.linspace(np.pi,-np.pi,res_size[0])
        lats = np.linspace(-np.pi/2,np.pi/2,res_size[1])
        X = np.arange(res_size[0])
        Y = np.arange(res_size[1])
        XX, YY = np.meshgrid(X,Y) 
        longs2, lats2 = np.meshgrid(longs,lats)
        Px = np.multiply(np.cos(lats2),np.cos(longs2))
        Pz = np.multiply(np.cos(lats2),np.sin(longs2))
        Py = np.sin(lats2)
        YY_flat = YY.flatten()
        XX_flat = XX.flatten()
        Px_flat = Px.flatten()
        Py_flat = Py.flatten()
        Pz_flat = Pz.flatten()
        P = np.vstack([XX_flat,YY_flat]).T
        Ps = np.vstack([Px_flat,Py_flat,Pz_flat]).T
        
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(Ps)
        o3d.visualization.draw_geometries([pcd])
        """

        """
        theta = np.arctan2(Pz_flat,Px_flat)
        print(theta.shape)
        phi = np.arctan(np.sqrt(np.square(Px_flat)+np.square(Pz_flat)),Py_flat)
        print(phi.shape)
        r = res_size[1]*phi/ ((180 * np.pi ) / 180)
        print(r.shape)
        
        ImPoints_int = np.array([0.5 * res_size[0] + r*np.cos(theta),
                    0.5 * res_size[1] + r*np.sin(theta)],int).T
        
        print(np.min(phi),np.max(phi))
        """
        Ps = np.expand_dims(Ps, -2)
        
        if rvec is None:
            rvec = np.array([0., 0., 0.])
        tvec = np.array([0., 0., 0.])
        
        #ImPoints, _ = cv2.fisheye.projectPoints(Ps, rvec, tvec, K, D)
        ImPoints, _ = cv2.omnidir.projectPoints(Ps,rvec, tvec, K,1, D)
        
        ImPoints_int = (ImPoints[:,0,:]).astype(int)
        ImPoints_int = np.maximum(ImPoints_int, [0,0])
        ImPoints_int = np.minimum(ImPoints_int, [img.shape[1]-1,img.shape[0]-1])
        ImPoints_int[self.mask[ImPoints_int[:,1],ImPoints_int[:,0]]==0] = 0
        #masked = np.where(mask[ImPoints_int[:,0],ImPoints_int[:,1]]==255, True, False)
        
        #ImPoints_inMask = ImPoints[masked]
        
        #ImPoints_inMask_int = ImPoints_inMask.astype(int)
        res = np.zeros((res_size[1],res_size[0],3),np.uint8)
        res[P[:,1],P[:,0]] = img[ImPoints_int[:,1],ImPoints_int[:,0],:]
        #res[ImPoints_int[:,0],ImPoints_int[:,1]] = img[ImPoints_int[:,0],ImPoints_int[:,1],:]
        
        
        #res[P[:,1],P[:,0]] = (np.tile(Pz.flatten(), (3,1)).T - mini)/(maxi-mini)*255
        
        dist2centerx = np.abs(img.shape[1]//2 - ImPoints_int[:,0])
        dist2centerx = dist2centerx.reshape((res_size[1],res_size[0]))
        
        return res, dist2centerx
        
    def createHemisphericalSelf(self, img):
        K = self.K
        D = self.D
        return self.createHemispherical(K, D, img)

    def createHemispherical(self, K,D, img, res_size=None):
        if res_size == None:
            res_size = (img.shape[1],img.shape[0])
        R = min(res_size)

        #deal with res image pixels
        X = np.arange(R)
        Y = np.arange(R)
        XX, YY = np.meshgrid(X,Y) 
        YY_flat = YY.flatten()
        XX_flat = XX.flatten()
        P = np.vstack([XX_flat,YY_flat]).T

        #angle on z axis
        XX_norm = XX * 2 / R - 1
        YY_norm = YY * 2 / R - 1
        radius = np.linalg.norm([XX_norm,YY_norm],axis=0)
        phi = np.arcsin(radius)

        Px = XX_norm
        Py = YY_norm
        Pz = np.cos(phi)

        #rearrange for cv2 function
        Px_flat = Px.flatten()
        Py_flat = Py.flatten()
        Pz_flat = Pz.flatten()
        Ps = np.vstack([Px_flat,Py_flat,Pz_flat]).T

        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(Ps)
        pcd.colors = o3d.utility.Vector3dVector(Ps*255)
        o3d.visualization.draw_geometries([pcd])
        """


        Ps = np.expand_dims(Ps, -2)
        
        #project points on original image
        rvec = np.array([0., 0., 0.])
        tvec = np.array([0., 0., 0.])

        
        ImPoints, _ = cv2.omnidir.projectPoints(Ps,rvec, tvec, K,1, D)

        #crop points to image rang int
        ImPoints_int = (ImPoints[:,0,:]).astype(int)
        ImPoints_int = np.maximum(ImPoints_int, [0,0])
        ImPoints_int = np.minimum(ImPoints_int, [img.shape[1]-1,img.shape[0]-1])

        
        
    
        #generate res image
        res = np.zeros((R,R,3),np.float32)
        res[P[:,1],P[:,0]] = img[ImPoints_int[:,1],ImPoints_int[:,0],:]
        

        #res, _ = rescale(res, 1000)
        #display(res)
        return res

    def testRay(self, img):
        size = img.shape
        P = np.array([[[size[1]*0.15,size[0]/2]]]).astype(np.float32)
        xi = np.array([1]).astype(np.float32)

        rvec = np.array([0., 0., 0.])
        points = cv2.omnidir.undistortPoints(P,self.K,self.D,xi,rvec)
        print(points)

        img = cv2.circle(img, (int(P[0,0,0]),int(P[0,0,1])), 10, (0,0,255),-1)
        display(img)

        return None

    def testUndistort(self, img):
        xi = np.array([1]).astype(np.float32)
        rvec = np.array([0., 0., 0.])
        #rvec = np.array([0., 0., 0])

        #res = cv2.omnidir.undistortImage(img, self.K, self.D, xi, cv2.omnidir.RECTIFY_PERSPECTIVE,R = rvec)
        res = cv2.omnidir.undistortImage(img, self.K, self.D, xi, cv2.omnidir.RECTIFY_LONGLATI,R = rvec)
        display(res)
        return res

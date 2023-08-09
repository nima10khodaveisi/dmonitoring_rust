import cv2
import glob
import random
import numpy as np

CHESS_PATH = "Images_From_Interior_Camera/34mm_CheckerBoard/"

obj_points = []
img_points = []

CHECKBOARD = (4, 6)

images = glob.glob(CHESS_PATH + "*.png")

objp = np.zeros((CHECKBOARD[0] * CHECKBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKBOARD[0], 0:CHECKBOARD[1]].T.reshape(-1, 2)

cnt = 0

for fname in images: 
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("this is gray size ", gray.shape[::-1])


    ret, corners = cv2.findChessboardCorners(gray, CHECKBOARD)

    if ret == True:
        cnt = cnt + 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners2)

        # gray = cv2.drawChessboardCorners(gray, (4, 6), corners2, ret)
        # cv2.imshow('gray', gray)
        # cv2.waitKey(500)

    
# cv2.destroyAllWindows()

print("this is cnt: ", cnt, gray.shape[::-1])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("what is ret though? ", ret)

print("Camera matrix : \n")
print(mtx)
print("+++++++++++++++")
print(np.array(mtx).reshape(3, 3))
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)


print("==============================================")

cap = cv2.VideoCapture("interior_center_day.mkv")
ret, frame = cap.read()
# frame = cv2.imread(CHESS_PATH + "50cm_Middle_1Box.png")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

height, width = frame.shape[:2]

print(frame.shape)

mean_error = 0
for i in range(len(obj_points)):
 imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
 error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
 mean_error += error
print( "total error: {}".format(mean_error/len(obj_points)) )

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.array(mtx, dtype='float32').reshape(3, 3), dist, (width, height), 1, (width, height))
print("this is new camera matrix")
print(newcameramtx)
dst = cv2.undistort(gray, mtx, dist, None, newcameramtx)
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width,height), 5)
# dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]

cv2.imshow('undistort', dst)
cv2.waitKey(0)

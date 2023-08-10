import cv2
import glob
import numpy as np

CHESS_PATH = "Images_From_Interior_Camera/34mm_CheckerBoard/"

sh = (1138, 1516)

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])

def calib_camera(recalc=False):
    global ret, mtx, dist, rvecs, tvecs, sh

    if not recalc: 
        mtx = np.load('mtx.npy')
        dist = np.load('dist.npy')
        rvecs = np.load('rvecs.npy')
        tvecs = np.load('tvecs.npy')
        return 0, mtx, dist, rvecs, tvecs


    obj_points = []
    img_points = []

    CHECKBOARD = (4, 7)

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
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners2)

            # gray = cv2.drawChessboardCorners(gray, (4, 6), corners2, ret)
            # cv2.imshow('gray', gray)
            # cv2.waitKey(500)

        
    # cv2.destroyAllWindows()

    print("this is cnt: ", cnt, gray.shape[::-1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    sh = gray.shape[:2]

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

    np.save('mtx', mtx)
    np.save('dist', dist)
    np.save('rvecs', rvecs)
    np.save('tvecs', tvecs)

    return ret, mtx, dist, rvecs, tvecs


def undist_frame(frame, sz=(1440, 960)):
    calib_camera(False)
    mtx = np.load('mtx.npy')
    dist = np.load('dist.npy')
    rvecs = np.load('rvecs.npy')
    tvecs = np.load('tvecs.npy')

    print("==============================================")

    # frame = cv2.imread(CHESS_PATH + "50cm_Middle_1Box.png")
    print("this is frame shape", frame.shape)
    # gray = cv2.cvtColoir(frame, cv2.COLOR_BGR2GRAY)

    height, width = frame.shape[:2]

    mtx = np.array(mtx, dtype='float32').reshape(3, 3)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (sh[1], sh[0]), 0.4, sz)
    print("this is new camera matrix")
    print(newcameramtx)
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width,height), 5)
    # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    print("this is roi ", roi)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    # cv2.imshow('undistort', dst)
    # cv2.waitKey(0)
    return dst, newcameramtx


def find_calib(frame, recalc=False):
    if recalc == True:
        calib_camera(False)
        img1 = cv2.imread(CHESS_PATH + "50cm_Right_0Box.png")
        img2 = cv2.imread(CHESS_PATH + "50cm_Middle_1Box.png")
        img1, _ = undist_frame(img1, (img1.shape[1], img1.shape[0]))
        img2, _ = undist_frame(img2, (img2.shape[1], img2.shape[0]))

        dst, K = undist_frame(frame, (frame.shape[1], frame.shape[0]))
        
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(descriptors1, descriptors2)

        matches = sorted(matches, key=lambda x: x.distance)

        num_matches_to_keep = 100
        good_matches = matches[:num_matches_to_keep]
        matched_pts1 = np.array([keypoints1[match.queryIdx].pt for match in good_matches])
        matched_pts2 = np.array([keypoints2[match.trainIdx].pt for match in good_matches])

        
        # why these lines broke the recoverPose function?
        # matched_pts1 = np.c_[matched_pts1[:, :2], np.ones((matched_pts1.shape[0], 1), dtype=np.float32)]
        # matched_pts2 = np.c_[matched_pts2[:, :2], np.ones((matched_pts2.shape[0], 1), dtype=np.float32)]

        # matches_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)
        # cv2.imshow('Matches', matches_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        F, mask = cv2.findFundamentalMat(matched_pts1, matched_pts2, cv2.FM_RANSAC)

        print(matched_pts1.shape, matched_pts2.shape, matched_pts2.dtype, matched_pts1.dtype)

        print("this is F: ", F)

        K = K.astype(np.float64)

        # E = np.dot(np.dot(K.T, F), K)
        print(matched_pts1)
        E, mask = cv2.findEssentialMat(matched_pts1, matched_pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0) # is this different?
        print("this is E: ", E)
        print(E.shape, E.dtype)
        _, R, t, _ = cv2.recoverPose(E=E, points1=matched_pts1, points2=matched_pts2, cameraMatrix=K)
        np.save('R', R)

    R = np.load('R.npy')

    euler_angles = rotationMatrixToEulerAngles(R)

# Convert radians to degrees
    euler_angles_deg = np.degrees(euler_angles)

    yaw = euler_angles_deg[2]
    pitch = euler_angles_deg[0]
    roll = euler_angles_deg[1]

    print("angles are", yaw, pitch, roll)

    return np.array([roll, pitch, yaw], dtype='float32').reshape(1, 3)
    

if __name__ == "__main__":
    calib_camera(False) # change this to True to recalc the matrices 
    # F = find_fund_mat()
    cap = cv2.VideoCapture('samples/interior_center-0208.mkv')
    _, frame = cap.read()
    frame = cv2.resize(frame, (1440, 960))
    find_calib(frame, True)
    # frame = cv2.imread(CHESS_PATH + "50cm_Middle_1Box.png")
    # dst, F = undist_frame(frame)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)

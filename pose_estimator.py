import cv2
import numpy as np
import json 
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class PoseEstimate(Node):
    def __init__(self):
        super().__init__('detection_node')
    
        # Load intrinsics
        calibration_path = "../../../../systems/SVM/birdseye/fisheye/intrinsics/22611253.json"

        with open(calibration_path, 'r') as json_file:
            calibration_params = json.load(json_file)

        self.D = np.array(calibration_params["distortion_coefficients"])
        self.K = np.array(calibration_params["camera_matrix"])

        self.fisheye = True
        self.ros = True

        # Board paramters
        self.n_rows = 9
        self.n_cols = 6 
        self.square_size = 0.03 # m

        # Initialize object points in the real world
        self.objp = np.zeros((self.n_rows * self.n_cols, 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.n_rows, 0:self.n_cols].T.reshape(-1, 2)  # Define the coordinates of a flat checkerboard in 3D (Z=0)
        self.objp = self.square_size * self.objp

        if self.ros:
            self.subscription = self.create_subscription(Image, '/blackfly_1/image_raw', self.image_callback, 10)
            rclpy.spin(self)
        else:
            video_path = '/home/gilbertogonzalez/Downloads/IMG_0447.mov'
            # Open the video file
            self.cap = cv2.VideoCapture(video_path)
            self.estimate_vid()

    def image_callback(self, msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        buf = buf.reshape(msg.height, msg.width, 1)
        cv_img = cv2.cvtColor(buf, cv2.COLOR_BAYER_RG2RGB)
        cv_image = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)
        

        self.estimate(cv_image)

    def estimate(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Arrays to store object points and image points
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        # Detect corners in the image
        start = time.time()
        ret, corners = cv2.findChessboardCorners(gray, (self.n_rows, self.n_cols), flags=cv2.CALIB_CB_FAST_CHECK)
        end = time.time()
        print(f"time: {end-start}")
        if ret:
            corner_size = 11
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
            corners = cv2.cornerSubPix(gray, corners, (corner_size, corner_size), (-1, -1), criteria)

            obj_points.append(self.objp)
            img_points.append(corners)
        else:
            print(f"ret: {ret}")
            self.show(img)
            return

        img_points = np.array(img_points, dtype=np.float32).squeeze()
        img_points = np.expand_dims(img_points, 1)

        obj_points = np.array(obj_points, dtype=np.float32).squeeze()

        undistorted = cv2.fisheye.undistortPoints(img_points, self.K, self.D)
        ret_pnp, rvec, tvec = cv2.solvePnP(obj_points, undistorted, np.eye(3), np.zeros((1,5)))

        # print(f"\nRmtx: {R}")
        # print(f"tvec: {tvec}")

        # Draw and show
        if ret_pnp:
            # cv2.drawChessboardCorners(img, (self.n_rows,self.n_cols), corners, ret)
            img_draw = cv2.drawFrameAxes(img, self.K, self.D, rvec, tvec, 0.2, 5)
            self.show(img_draw)
        else:
            print(f"ret_pnp: {ret_pnp}")
            return
        
    def estimate_vid(self):
        while True:
            # Read a frame from the video
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Arrays to store object points and image points
            obj_points = []  # 3D points in real world space
            img_points = []  # 2D points in image plane

            # Detect corners in the image
            start = time.time()
            ret, corners = cv2.findChessboardCorners(gray, (self.n_rows, self.n_cols), flags=cv2.CALIB_CB_FAST_CHECK)
            end = time.time()
            print(f"time: {end-start}")
            if ret:
                corner_size = 11
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
                corners = cv2.cornerSubPix(gray, corners, (corner_size, corner_size), (-1, -1), criteria)

                obj_points.append(self.objp)
                img_points.append(corners)

                img_points = np.array(img_points, dtype=np.float32).squeeze()
                img_points = np.expand_dims(img_points, 1)

                obj_points = np.array(obj_points, dtype=np.float32).squeeze()

                ret_pnp, rvec, tvec = cv2.solvePnP(obj_points, img_points, np.eye(3), np.zeros((1,5)), flags=cv2.SOLVEPNP_ITERATIVE)


                # print(f"\nRmtx: {R}")
                # print(f"tvec: {tvec}")

                # Draw and show
                
                if ret_pnp:
                    cv2.drawChessboardCorners(img, (self.n_rows,self.n_cols), corners, ret)
                    img_draw = cv2.drawFrameAxes(img, np.eye(3), np.zeros((1,5)), rvec, tvec, 0.2, 5)
                    self.show(img)
                else:
                    print(f"ret_pnp: {ret_pnp}")
                    # return
            else:
                print(f"ret: {ret}")
                self.show(img)
                # return

            

    def show(self, img):
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
        cv2.imshow('img', img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    PoseEstimate()

if __name__ == '__main__':
    main()

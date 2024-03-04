#!/usr/bin/env python3

import numpy as np
import cv2
import json

def projpts(pts, P):
    projected = P.dot(pts)
    w = projected[2, :]
    projected /= w

    return projected

def draw_axes(frame, R, t, mtx, dist, line_length):
    xyzo = np.float32([[line_length, 0, 0], [0, line_length, 0], [0, 0, -line_length], [0, 0, 0]]).reshape(-1,3)

    axis, _ = cv2.projectPoints(xyzo, R, t, mtx, dist)
    axis = axis.astype(int)
    cv2.line(frame, tuple(axis[3].ravel()),
                    tuple(axis[0].ravel()), (255, 0, 0), 3)
    cv2.line(frame, tuple(axis[3].ravel()),
                    tuple(axis[1].ravel()), (0, 255, 0), 3)
    cv2.line(frame, tuple(axis[3].ravel()),
                    tuple(axis[2].ravel()), (0, 0, 255), 3)

def draw_cube(frame, P, x, y, z, w, h, color):
    pts = np.float32([
        [x, y, -z, 1], [x + w, y, -z, 1], [x + w, y + w, -z, 1], [x, y + w, -z, 1],
        [x, y, -(z + h), 1], [x + w, y, -(z + h), 1],
        [x + w, y + w, -(z + h), 1], [x, y + w, -(z + h), 1]
    ]).reshape(-1, 4).transpose()

    projected = projpts(pts, P)
    under = projected[:, :4]
    over = projected[:, 4:]

    under_fill = np.array([[int(under[0, i]), int(under[1, i])] for i in range(under.shape[1])])
    cv2.fillPoly(frame, pts = [under_fill], color=color)

    [cv2.line(frame, (int(under[0, i]), int(under[1, i])),
                     (int(under[0, i - 1]), int(under[1, i - 1])), (0, 255, 0), 2)
        for i in range(under.shape[1])]

    [cv2.line(frame, (int(over[0, i]), int(over[1, i])),
                     (int(over[0, i - 1]), int(over[1, i - 1])), (255, 0, 0), 2)
        for i in range(over.shape[1])]

    [cv2.line(frame, (int(under[0, i]), int(under[1, i])),
                     (int(over[0, i]), int(over[1, i])), (0, 0, 255), 2)
        for i in range(under.shape[1])]
    
if __name__ == '__main__':
    # Calibration step
    patternsize = (9, 6)

    # Load intrinsics
    calibration_path = "calibration/iphone12.json"
    with open(calibration_path, 'r') as json_file:
        calibration_params = json.load(json_file)
    dist = np.array(calibration_params["distortion_coefficients"])
    mtx = np.array(calibration_params["camera_matrix"])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    r, c = patternsize
    objp = np.zeros((r * c, 3), np.float32)
    objp[:, :2] = np.mgrid[0:r, 0:c].T.reshape(-1, 2)

    video_path = 'test_assets/IMG_0447.mov'
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, img = cap.read()
        # img = cv2.imread("/test_assets/img.png")

        blank_img = np.zeros((img.shape), dtype=np.uint8)

        key = cv2.waitKey(1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected, corners = cv2.findChessboardCorners(gray, patternsize, flags=cv2.CALIB_CB_FAST_CHECK)

        if detected:
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            _, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
            R, _ = cv2.Rodrigues(rvec)
            P = mtx.dot(np.hstack([R, tvec]))

            x = y = z = 0
            draw_axes(blank_img, R, tvec, mtx, dist, 2)
            for i in range(0, patternsize[0] - 1, 2):
                for j in range(0, patternsize[1] - 1, 2):
                        draw_cube(blank_img, P, (x + i), (y + j), z, 1, 1, (10, 10, 100))
        
        cv2.imshow('img', blank_img)

        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()

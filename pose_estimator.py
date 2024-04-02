#!/usr/bin/env python3

import numpy as np
import cv2
import json
import argparse

def projpts(pts, P):
    projected = P.dot(pts)
    w = projected[2, :]
    projected /= w

    return projected

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
    
    return frame
    
if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--debug', type=bool, default=False, help="'--debug = True' for vizualizing corner points")
    args = parser.parse_args()

    # Checkerboard params
    pattern = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

    # Load intrinsics
    calibration_path = "calibration/iphone12.json"
    with open(calibration_path, 'r') as json_file:
        calibration_params = json.load(json_file)
    D = np.array(calibration_params["distortion_coefficients"])
    K = np.array(calibration_params["camera_matrix"])
    
    r, c = pattern
    objp = np.zeros((r * c, 3), np.float32)
    objp[:, :2] = np.mgrid[0:r, 0:c].T.reshape(-1, 2)

    # Open video file
    video_path = 'test_assets/IMG_0447.mov'
    cap = cv2.VideoCapture(video_path)

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                print("\nAll done, goodbye!")
                break
            # img = cv2.imread("/test_assets/img.png")

            key = cv2.waitKey(1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_FAST_CHECK)
            

            if ret:
                cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                # print(f"{corners[r-1][0].tolist() = }")
                _, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
                R, _ = cv2.Rodrigues(rvec)
                P = K.dot(np.hstack([R, tvec]))

                src_pts = np.float32([corners[0][0].tolist(), corners[r-1][0].tolist(), corners[(r*c)-r][0].tolist(), corners[(r*c)-1][0].tolist()])

                # Computing dst
                scale = 50
                scaled_r, scaled_c = r*scale, c*scale
                t_x = img.shape[1]/2 - (scaled_r / 2)
                t_y = img.shape[0]/2 - (scaled_c / 2)
                dst_pts = np.float32([[scaled_r+t_x, scaled_c+t_y], [t_x, scaled_c+t_y], [scaled_r+t_x, t_y], [t_x, t_y]])

                if args.debug:
                    # Draw circles around each point
                    for i, pt in enumerate(src_pts):
                        pt = tuple(map(int, pt)) 
                        cv2.circle(img, pt, 10, (0, 0, 255), -1)
                        cv2.putText(img, str(i), pt, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

                    for i, pt in enumerate(dst_pts):
                        pt = tuple(map(int, pt))
                        cv2.circle(img, pt, 10, (0, 255, 0), -1)
                        cv2.putText(img, str(i), (pt[0]+40, pt[1]+40), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                proj_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
                # cv2.drawFrameAxes(proj_img, K, D, rvec, tvec, 3.0)

                x = y = z = 0
                blank_img = np.zeros((img.shape), dtype=np.uint8)
                for i in range(0, pattern[0] - 1, 2):
                    for j in range(0, pattern[1] - 1, 2):
                            pose_img = draw_cube(blank_img, P, (x + i), (y + j), z, 1, 1, (10, 10, 100))

                welcome_img = cv2.putText(np.zeros((img.shape), dtype=np.uint8), "Welcome!", (img.shape[1]//2-200, img.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4, cv2.LINE_AA)
                proj_img    = cv2.putText(proj_img, "proj_img", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                pose_img    = cv2.putText(pose_img, "pose_img", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                img         = cv2.putText(img, "original_img", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(proj_img, "No corners detected", (10, proj_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(pose_img, "No corners detected", (10, pose_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, "No corners detected", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                
            # Resizing
            img_scale = 0.35
            resize1 = cv2.resize(proj_img, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_AREA)
            resize2 = cv2.resize(pose_img, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_AREA)
            resize3 = cv2.resize(img, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_AREA)
            resize4 = cv2.resize(welcome_img, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_AREA)

            # Concat imgs together for better viz
            half1  = cv2.hconcat([resize3, resize2]) 
            half2  = cv2.hconcat([resize4, resize1]) 
            result = cv2.vconcat([half2, half1]) 

            cv2.imshow('result', result)

            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    except KeyboardInterrupt:
        print("\nWhy'd you ctrl-c?? The vid wasnt over! Jk thanks for watchin")

    cap.release()
    cv2.destroyAllWindows()

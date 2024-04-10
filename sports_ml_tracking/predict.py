#!/usr/bin/env python3

import numpy as np
import cv2
import math

import torch
from ultralytics import YOLO

from kalmanfilter import KalmanFilter 

## TODO:
# - understand kalman filter
# - organize kalman filter code
# - dockerize
# - put into its own repo

kalman_predict = True
polynomial_predict = True

# Kalman filter params
fps = 70
dt = 1/fps
A = np.eye(4)
A[0, 2] = dt
A[1, 3] = dt

# gravity control
u = np.array([0, 50])
B = np.zeros((4, 2))
B[0, 0] = dt**2/2
B[1, 1] = dt**2/2
B[2, 0] = dt
B[3, 1] = dt

# x, y, vx, vy
mu = np.array([0,0,0,0])
P = np.diag([10,10,10,10])**2
res=[]

data = {'ball':[],
        'rim':[]}

x_list = []
y_list = []

def detect(cv_image, thresh=0.5):
    results = model(cv_image, conf=thresh, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            bbox = box.xyxy[0]
            x1, y1, x2, y2 = bbox
            c = box.cls

            if c == 0:
                data['ball'] += [(int((x1 + x2) / 2), int((y1 + y2) / 2))] # center of ball
            if c == 1:
                data['rim'] += [((x1, y1), (x2, y1))] # top-left and top-right of rim

    return results

if __name__ == '__main__':
    model = YOLO('best.pt')

    count = 0

    # Set GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu") 

    model.to(device=device)

    video_path = "test_assets/steph.mov"
    # video_path = "/home/gilberto/Downloads/test_imgs/steph2.mov"
    # video_path = "/home/gilberto/Downloads/test_imgs/klay.mov"

    vid = cv2.VideoCapture(video_path)

    while True:        
        success, frame = vid.read()
        kf = KalmanFilter()

        if success:
            results = detect(frame)

            annotated_frame = results[0].plot()

            if len(data['ball']) > 0 and len(data['rim']) > 0:
                ball_x = data['ball'][-1][0]
                ball_y = data['ball'][-1][1]
                rim_x1 = data['rim'][-1][0][0]
                rim_x2 = data['rim'][-1][1][0]
                rim_y  = data['rim'][-1][0][1]

                x_list.append(ball_x)
                y_list.append(ball_y)
                
                # Kalman filter
                if kalman_predict:
                    predicted, mu, statePost, errorCovPre = kf.predict(int(ball_x), int(ball_y))
                    mu,P = kf.kal(mu,P,B,u,z=None)
                
                    res += [(mu,P)]
                    mu2 = mu
                    P2 = P
                    res2 = []

                    for _ in range(fps*2):
                        mu2,P2 = kf.kal(mu2,P2,B,u,z=None)
                        res2 += [(mu2,P2)]

                    
                    xe = [mu[0] for mu,_ in res]
                    xu = [2*np.sqrt(P[0,0]) for _,P in res]
                    ye = [mu[1] for mu,_ in res]
                    yu = [2*np.sqrt(P[1,1]) for _,P in res]
                    
                    xp=[mu2[0] for mu2,_ in res2] 
                    yp=[mu2[1] for mu2,_ in res2]

                    xpu = [np.sqrt(P[0,0]) for _,P in res2]
                    ypu = [np.sqrt(P[1,1]) for _,P in res2]

                    # # Draw predicted line
                    # for n in range(len(xp)):
                    #     cv2.circle(annotated_frame, (int(xp[n]),int(yp[n])), 5, (255, 0, 255), -1)
                        
                    if count > 0:
                        Ak, Bk, Ck = np.polyfit(xp, yp, 2)
                        for x in range(frame.shape[1]):
                            y = int(Ak * x ** 2 + Bk * x + Ck)
                            cv2.circle(annotated_frame,(x, y), 5, (255, 0, 255), -1)

                        a = Ak
                        b = Bk
                        c = Ck - rim_y

                        d = b*b - 4*a*c # discriminant
                        if d >= 0:
                            x1 = int((-b - math.sqrt(d)) / (2 * a)) # solution 1
                            x2 = int((-b + math.sqrt(d)) / (2 * a)) # solution 2

                        cv2.circle(annotated_frame,(int(rim_x1), int(rim_y)), 5, (0, 255, 0), 5)
                        cv2.circle(annotated_frame,(int(rim_x2), int(rim_y)), 5, (0, 255, 0), 5)
                        cv2.circle(annotated_frame,(x1, int(rim_y)), 5, (255, 255, 0), 5)
                        cv2.circle(annotated_frame,(x2, int(rim_y)), 5, (255, 255, 0), 5)

                        if int(rim_x1) < x1 < int(rim_x2) or int(rim_x1) < x2 < int(rim_x2):
                            cv2.putText(annotated_frame, "Basket", (int(annotated_frame.shape[1] - 250), 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                    
                    count += 1

                # Polynomial regression
                if polynomial_predict:
                    Ap, Bp, Cp = np.polyfit(x_list, y_list, 2)
                    for x in range(frame.shape[1]):
                        y = int(Ap * x ** 2 + Bp * x + Cp)
                        cv2.circle(annotated_frame,(x, y), 5, (0, 0, 255), -1)

                    a = Ap
                    b = Bp
                    c = Cp - rim_y

                    d = b*b - 4*a*c # discriminant
                    if d >= 0:
                        x1 = int((-b - math.sqrt(d)) / (2 * a)) # solution 1
                        x2 = int((-b + math.sqrt(d)) / (2 * a)) # solution 2

                    # Rim points
                    cv2.circle(annotated_frame,(int(rim_x1), int(rim_y)), 5, (0, 255, 0), 5)
                    cv2.circle(annotated_frame,(int(rim_x2), int(rim_y)), 5, (0, 255, 0), 5)
                    
                    # Ball point of intersection with rim_y
                    cv2.circle(annotated_frame,(x1, int(rim_y)), 5, (255, 255, 0), 5)
                    cv2.circle(annotated_frame,(x2, int(rim_y)), 5, (255, 255, 0), 5)

                    # Check both solutions
                    if int(rim_x1) < x1 < int(rim_x2) or int(rim_x1) < x2 < int(rim_x2):
                        cv2.putText(annotated_frame, "Basket", (int(annotated_frame.shape[1] - 250), 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
            
            else:
                continue

            # img_scale = 0.8
            # annotated_frame = cv2.resize(annotated_frame, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_AREA)

            cv2.imshow("annotated_frame", annotated_frame)

            key = cv2.waitKey(0) & 0xFF # show per frame
            #key = cv2.waitKey(1)
            #if key == ord(" "):  # Spacebar to pause
            #    cv2.waitKey(-1)

        else:
            break

    vid.release()
    cv2.destroyAllWindows()

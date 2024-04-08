#!/usr/bin/env python3

import numpy as np
import cv2
import math

import torch
from ultralytics import YOLO

from kalmanfilter import KalmanFilter 

## TODO:
# - continue training model
# - understand kalman filter
# - organize kalman filter code
# - why is polyfitting the kalman results so slow?
# - dockerize

kalman_predict = True
polynomial_predict = True

########################

noise = 3
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

########################

data = {'ball':[],
        'rim':[]}

x_list = []
y_list = []

def detect(cv_image, thresh=0.1):
    results = model(cv_image, conf=thresh, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            bbox = box.xyxy[0]
            x1, y1, x2, y2 = bbox
            c = box.cls

            if c == 0: # ball
                data['ball'] += [(int((x1 + x2) / 2), int((y1 + y2) / 2))] # center of ball
            if c == 1: # rim
                data['rim'] += [((x1, y1), (x2, y1))] # top-left and top-right of rim

    return results

if __name__ == '__main__':
    model = YOLO('best.pt')

    count = 0

    # plotted_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Set GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu") 

    model.to(device=device)

    # video_path = "test_assets/cropped.mp4"
    video_path = "/home/gilberto/Downloads/test_imgs/steph4.mov"
    # video_path = "/home/gilberto/Downloads/test_imgs/klay.mov"

    vid = cv2.VideoCapture(video_path)

    while True:
        plotted_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        success, frame = vid.read()
        kf = KalmanFilter()

        if success:
            results = detect(frame)

            # if len(data['rim']) > 0:
            #     print(f"{data['rim'][-1][0][0] = }")

            annotated_frame = results[0].plot()

            if len(data['ball']) > 0:
                cv2.circle(plotted_img, data['ball'][-1], 20, (0, 255, 0), 2)
                x_list.append(data['ball'][-1][0])
                y_list.append(data['ball'][-1][1])
                
                # Kalman filter
                if kalman_predict:
                    predicted, mu, statePost, errorCovPre = kf.predict(int(data['ball'][-1][0]), int(data['ball'][-1][1]))
                    mu,P = kf.kal(mu,P,B,u,z=None)
                
                    ##### Prediction #####
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

                    # # predicted next step
                    # cv2.circle(annotated_frame,(int(xe[-1]),int(ye[-1])),5,(255, 255, 0),-1)

                    # print(f"{xp = }")
                    if count > 1:
                        for n in range(len(xp)): # x e y predicha
                            # incertidumbreP=(xpu[n]+ypu[n])/25
                            cv2.circle(annotated_frame,(int(xp[n]),int(yp[n])),5,(255, 0, 255), -1)
                            # Ak, Bk, Ck = np.polyfit(xp, yp, 2)
                            # for x in range(((len(xp)) + 1000)):
                            #     y = int(Ak * x ** 2 + Bk * x + Ck)
                            #     cv2.circle(annotated_frame,(x, y), 5, (0, 0, 255), cv2.FILLED)

                            #     # if len(data['rim']) > 0:
                            #     a = Ak
                            #     b = Bk
                            #     c = Ck - data['rim'][-1][0][1]

                            #     discriminant = b ** 2 - 4 * a * c

                            #     # Check if the discriminant is non-negative
                            #     if discriminant >= 0:
                                
                            #         x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))

                            #         cv2.circle(annotated_frame,(x, int(data['rim'][-1][1][1])), 5, (255, 255, 0), 5)

                    
                    count += 1

                # Polynomial regression
                if polynomial_predict:
                    Ap, Bp, Cp = np.polyfit(x_list, y_list, 2)
                    for x in range(((len(x_list)) + 1000)):
                        y = int(Ap * x ** 2 + Bp * x + Cp)
                        cv2.circle(annotated_frame,(x, y), 5, (0, 0, 255), cv2.FILLED)

                    if len(data['rim']) > 0:
                        a = Ap
                        b = Bp
                        c = Cp - data['rim'][-1][0][1]

                        x = int((-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a))

                        # print(f"{data['rim'][-1][0][1] = }")
                        # print(f"{y = }")

                        cv2.circle(annotated_frame,(int(data['rim'][-1][0][0]), int(data['rim'][-1][0][1])), 5, (0, 255, 0), 5)
                        cv2.circle(annotated_frame,(int(data['rim'][-1][1][0]), int(data['rim'][-1][1][1])), 5, (0, 255, 0), 5)
                        cv2.circle(annotated_frame,(x, int(data['rim'][-1][1][1])), 5, (255, 255, 0), 5)

                        if int(data['rim'][-1][0][0]) < x < int(data['rim'][-1][1][0]):
                            cv2.putText(annotated_frame, "Basket", (int(annotated_frame.shape[1] - 250), 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
            
            else:
                continue

            # # Draw rim
            # if len(data['rim']) > 0:
            #     cv2.ellipse(plotted_img, data['rim'][-1], (45, 10), 0, 0, 360, (255, 255, 0), 2) 

            img_scale = 0.8
            resize1 = cv2.resize(annotated_frame, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_AREA)
            resize2 = cv2.resize(plotted_img, None, fx=img_scale, fy=img_scale, interpolation=cv2.INTER_AREA)

            cv2.imshow("annotated_frame", resize1)
            # cv2.imshow("plotter", resize2)

            # key = cv2.waitKey(0) & 0xFF # show per frame
            key = cv2.waitKey(1)
            if key == ord(" "):  # Spacebar to pause
                cv2.waitKey(-1)

        else:
            break

    vid.release()
    cv2.destroyAllWindows()
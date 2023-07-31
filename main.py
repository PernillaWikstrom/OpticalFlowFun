import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import numpy as np
import time

# Local package
import src as opticalFlowFun

parser = opticalFlowFun.get_parser()
orb_detector = opticalFlowFun.orbDetection(
    parser.nfeatures, opticalFlowFun.orb_parameters
)

# Read webcam
cap = cv.VideoCapture(0)
keypoints_trajectories = []
frame_idx = 0

while True:

    suc, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = frame.copy()

    if len(keypoints_trajectories) > 0:
        keypoints_trajectories = opticalFlowFun.lucas_kanade_optical_flow(
            keypoints_trajectories,
            img,
            previous_gray,
            frame_gray,
            opticalFlowFun.lk_params,
            parser.n_keypoints_trajectories,
        )

    # Update interval - When to update and detect new features
    if frame_idx % parser.detection_interval == 0:
        keypoints_trajectories, descriptors = orb_detector.get_features(
            keypoints_trajectories, frame_gray
        )

    frame_idx += 1
    previous_gray = frame_gray

    cv.imshow("Optical Flow", img)
    # Interupt window by pressing q
    if cv.waitKey(10) & 0xFF == ord("q"):
        break


cap.release()
cv.destroyAllWindows()

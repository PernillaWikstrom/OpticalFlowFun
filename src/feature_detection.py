import cv2 as cv
import numpy as np


class orbDetection:
    """
    Using orb-feature detection this class is creating a detector,
    and when asked it computes new keypoints and descriptors.
    """

    def __init__(self, n_features: int, params: dict):
        if n_features is not None:
            params["nfeatures"] = n_features
        self.detector = cv.ORB_create(**params)

    def get_features(self, keypoints_trajectories: list, frame_gray: np.ndarray):
        """
        Compute new keypoints and descriptors

        Args:
            keypoints_trajectories (list): list of detected keypoints trajectory
            frame_gray (np.ndarray): frame gray

        Returns:
            keypoints_trajectories, descriptors
        """
        mask = np.zeros_like(frame_gray)
        mask[:] = 255

        # Lastest point in latest found keypoints
        for x, y in [np.int32(trajectory[-1]) for trajectory in keypoints_trajectories]:
            cv.circle(mask, (x, y), 5, 0, -1)

        keypoints, descriptors = self.detector.detectAndCompute(frame_gray, mask=mask)
        if keypoints is not None:
            for kp in keypoints:
                keypoints_trajectories.append([(kp.pt[0], kp.pt[1])])
        return keypoints_trajectories, descriptors

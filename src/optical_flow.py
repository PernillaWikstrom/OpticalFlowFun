import cv2 as cv
import numpy as np


def lucas_kanade_optical_flow(
    keypoints_trajectories: list,
    img_RGB: np.ndarray,
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    lk_params: dict,
    n_keypoints_trajectories: int,
):
    """Compute sparse optical flow using lucas-kanade method

    Args:
        keypoints_trajectories (list): list of the detected keypoints trajectories
        img_RGB (np.ndarray):
        previous_frame (np.ndarray): gray frame
        current_frame (np.ndarray): gray frame
        lk_params (dict): lucas-kanade parameters
        n_keypoints_trajectories (int): how long to track each detected keypoint

    Returns:
        new_trajectories (list): updated list with respect to maximum
    """

    # Starting points
    p0 = np.float32([trajectory[-1] for trajectory in keypoints_trajectories]).reshape(
        -1, 1, 2
    )
    p1, _st, _err = cv.calcOpticalFlowPyrLK(
        previous_frame, current_frame, p0, None, **lk_params
    )
    p0r, _st, _err = cv.calcOpticalFlowPyrLK(
        current_frame, previous_frame, p1, None, **lk_params
    )
    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    good = d < 1

    new_trajectories = []
    # Get all detected keypoints
    for trajectory, (x, y), good_flag in zip(
        keypoints_trajectories, p1.reshape(-1, 2), good
    ):
        if not good_flag:
            continue
        trajectory.append((x, y))
        if len(trajectory) > n_keypoints_trajectories:
            del trajectory[0]
        new_trajectories.append(trajectory)
        # Newest detected point
        cv.circle(img_RGB, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Draw all the detected keypoints
    cv.polylines(
        img_RGB,
        [np.int32(trajectory) for trajectory in new_trajectories],
        False,
        (0, 255, 0),
    )
    cv.putText(
        img_RGB,
        "track count: %d" % len(new_trajectories),
        (20, 50),
        cv.FONT_HERSHEY_PLAIN,
        1,
        (0, 255, 0),
        2,
    )
    return new_trajectories

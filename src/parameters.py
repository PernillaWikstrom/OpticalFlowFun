import cv2 as cv


lk_params = dict(
    winSize=(15, 15),
    maxLevel=4,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
)


orb_parameters = dict(
    nfeatures=500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    patchSize=31,
)

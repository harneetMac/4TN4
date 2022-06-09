import cv2
import numpy as np
import matplotlib.pyplot as plt

# read given video
cap = cv2.VideoCapture('video1.mp4')
_, frame = cap.read()
oldGrayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frameHeight, frameWidth = oldGrayFrame.shape

numberFrames = 0

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=50,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(25, 25),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

mask = np.zeros_like(oldGrayFrame)
cv2.rectangle(mask, (frameWidth // 2, frameHeight // 2), (frameWidth, frameHeight), 255, -1)
# cv2.imshow("mask", mask)
# cv2.waitKey(0)

oldPts = cv2.goodFeaturesToTrack(oldGrayFrame, mask=mask, **feature_params)
mask = np.zeros_like(frame)
# cv2.imshow("newMask", mask)
# cv2.waitKey(0)

thickness = 35

while True:
    ret, frame = cap.read()
    if not ret:
        print(f'Reached end of frames')
        break

    print(f'{numberFrames=}')

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # optical flow
    newPts, status, error = cv2.calcOpticalFlowPyrLK(oldGrayFrame, frameGray, oldPts, None, **lk_params)

    # Select good points
    good_new = newPts[status == 1]
    good_old = oldPts[status == 1]

    thickness -= 0.95
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(np.int32)
        c, d = old.ravel().astype(np.int32)
        print(f'{i=}, {a=}, {b=}, {c=}, {d=}')
        mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), int(thickness*0.5))
        # mask = cv2.circle(mask, (a, b), int((thickness*0.5)), (0, 0, 255), -1)
        # frame = cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)
    img = cv2.add(frame, mask)

    # cv2.line(frame, (oldPts[0][0].astype(np.int32)), (newPts[0][0].astype(np.int32)), (0, 0, 255), thickness + 1)

    cv2.imshow("frame", img)

    if numberFrames == 33:
        cv2.imwrite("Last_Frame.jpg", img)
        print("Image Created")

    key = cv2.waitKey(1)
    if key == 27:
        break

    numberFrames += 1
    oldGrayFrame = frameGray.copy()
    oldPts = good_new.reshape(-1, 1, 2)
    # print(f'{oldPts=}')

print(f'{numberFrames= }')
cap.release()
cv2.destroyAllWindows()

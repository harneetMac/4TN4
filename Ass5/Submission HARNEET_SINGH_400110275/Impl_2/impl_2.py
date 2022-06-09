import cv2
import numpy as np

# read given images
img1Color = cv2.imread("image1_1.jpg")
img2Color = cv2.imread("image1_2.jpg")
img1 = cv2.cvtColor(img1Color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2Color, cv2.COLOR_BGR2GRAY)

# Initiate ORB detector
# ORB is similar to SIFT algorithm to detect features and keypoints
orb = cv2.ORB_create()

# find keypoints and descriptors in both images
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# use brute force method to match the keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
print(f'Total number of matches is {len(matches) = }')

# sort the matched based on the distance between the matches of the images
# shorter distance represents a better match, therefore it will allow us
# to pick top n matches
matches = sorted(matches, key=lambda x: x.distance)

# iterate through the match's distance
print("Distance between matches is:")
for m in matches:
    print(m.distance, end=', ')

# displaying top 20 matches on the images
matchedImages = cv2.drawMatches(img1Color, kp1, img2Color, kp2, matches[:20], None, flags=2)

# cv2.imshow("image 1", img1)
# cv2.imshow("image 2", img2)
cv2.imshow("matched", matchedImages)
cv2.imwrite("impl_2_result.jpg", matchedImages)
cv2.waitKey(0)

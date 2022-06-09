import cv2
import numpy as np

# read given image
img = cv2.imread("chess.png")
print(f'Given image size is {img.shape}')
height, width, _ = img.shape

# creating enlarged image of the given image for white background requirement
imgEnlarge = np.full((height+150, width+150, 3), 255, dtype='uint8')
imgEnlarge[50:50+height, 50:50+width] = img

# find the co-ordinates of the chess board to perform warp
# used trial and error to find the correct pixel values (and Paint app)
topLeftPt, topRightPt = [460, 120], [900, 350]
bottomLeftPt, bottomRightPt = [15, 310], [440, 775]

# using the points in image A to transform the image
inputPts = np.float32([topLeftPt, topRightPt, bottomLeftPt, bottomRightPt])

# creating output points to tell function where to warp the image
outputPts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

# use cv2's perspective transform function
transformationMat = cv2.getPerspectiveTransform(inputPts, outputPts)
outImg = cv2.warpPerspective(imgEnlarge, transformationMat, (width, height))

# drawing points on the image to show the corners
for x, y in inputPts:
    cv2.circle(imgEnlarge, (int(x), int(y)), 5, (0, 0, 255), cv2.FILLED)

cv2.imshow("chess image", imgEnlarge)
cv2.imshow("transformed image", outImg)
# cv2.imwrite("impl_1_input_points.png", imgEnlarge)
# cv2.imwrite("impl_1_result.png", outImg)
cv2.waitKey(0)

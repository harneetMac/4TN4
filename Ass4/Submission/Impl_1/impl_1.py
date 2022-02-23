import numpy as np
import cv2
from matplotlib import pyplot as plt

# creating image array
img = cv2.imread('3.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = grayImg.shape
imgSize = height * width

# applying filter to smooth the image - remove high frequency noise
gaussianFilteredImg = cv2.GaussianBlur(grayImg, (3,3), 0)

# binarizing the image using threshold function of OpenCV
# threshold value of 127 is chosen to find the black letters in the licence plate
# OTSU finds the optimal threshold value
thresVal, binImg = cv2.threshold(gaussianFilteredImg, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(f'Threshold Value: {thresVal}')
# cv2.imshow("binarized img", binImg)
# cv2.imwrite("impl_1_binarized_img.png", binImg)

# cv2's Open function (erode and then dilate) to remove small blobs
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
openedImg = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, open_kernel, iterations=3)

# cv2.imshow("opened img", openedImg)
# cv2.imwrite("impl_1_opened_img.png", openedImg)

'''
using contours to join all the continuous points along the boundary with approximation,
this contour function identifies white objects on black background,
cv2.CHAIN_APPROX_SIMPLE removes redundant points from the boundary and compresses the contour,
using underscore for hierarchy data because it is not needed in this case,
2nd tuple in the contours array stores the hierarchy but we do not need it in this case.
'''
contours, _ = cv2.findContours(openedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(grayImg, contours[0], -1, (0,255,0), 3)
# cv2.imshow("contours img", grayImg)
# cv2.imwrite('impl_1_contour_img.png', grayImg)

# iterating through each contour to get the area of the contour and find the license plate
# based on the area and height to width ratio in terms of full image size
for cont in contours:
    x, y, w, h = cv2.boundingRect(cont)
    area = cv2.contourArea(cont)
    print(f'w: {w}, h: {h}, area:{area}')

    if 2 <= w/h <= 2.5 and 7 < imgSize/area < 9:
        print(f'ROI - w: {w}, h: {h}, area:{area}')
        ROI = np.ones_like(grayImg)
        ROI[y:y+h, x:x+w] = openedImg[y:y+h, x:x+w]

cv2.imshow("img", ROI)
# cv2.imwrite('impl_1_extracted_plate.png', ROI)
cv2.waitKey(0)


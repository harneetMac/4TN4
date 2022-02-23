import numpy as np
import cv2
from matplotlib import pyplot as plt

# creating image array
img = cv2.imread('3.png', cv2.IMREAD_GRAYSCALE)

# applying filter to smooth the image
img = cv2.GaussianBlur(img, (5,5), 0)

# binarizing the image using threshold function of OpenCV
# threshold value of 127 is choosen to find the black letters in the licence plate
# OTSU finds the optimal threshold value
thresVal, binImg = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(thresVal)
# cv2.imshow("binarized img", binImg)
# plt.imshow(binImg, cmap='gray', vmin=0, vmax=255)
# plt.show()
# cv2.waitKey(0)

# cv2's Open function (erode and then dilate) to check the functionality of my logic
open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
openedImg = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, open_kernel, iterations=1)

close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
close = cv2.morphologyEx(openedImg, cv2.MORPH_CLOSE, close_kernel, iterations=1)
# cv2.imshow("opened img", close)
# # plt.imshow(close, cmap='gray', vmin=0, vmax=255)
# # plt.show()
# cv2.waitKey(0)

cnts = cv2.findContours(openedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(cnts)
cnt = cnts[0] if len(cnts) == 2 else cnts[1]
# print(cnt)

for c in cnt:
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    print(area)
    if area > 40000:
        ROI = np.ones_like(img)
        ROI[y:y+h, x:x+w] = close[y:y+h, x:x+w]
        ROI = cv2.GaussianBlur(ROI, (3,3), 0)
        # cv2.rectangle(close, (x, y), (x + w, y + h), (255, 255, 255), 2)

cv2.imshow("img", ROI)
cv2.waitKey(0)


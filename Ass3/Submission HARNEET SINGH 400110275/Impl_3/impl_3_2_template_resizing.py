import numpy as np
import cv2
from matplotlib import pyplot as plt

# read the template as gray scale
template = cv2.imread('circle.bmp', cv2.IMREAD_GRAYSCALE)

# read the image as gray scale and scale it up because we will be using
# the scaled up version (double size)
img = cv2.imread('messi.jpg', cv2.IMREAD_GRAYSCALE)
resizedImg = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cannyImg = cv2.Canny(resizedImg, 40, 140, L2gradient=True)

img2 = img.copy()
img = resizedImg
width, height = img.shape[::-1]

# increment the template size by 10% percent up to 200%
percentSpacing = 0.1
scalingFactor = np.arange(1, 2.1, percentSpacing)

# loop through all the scale levels to find the best match
for index, scale in enumerate(scalingFactor):
    resizeTemp = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    w, h = resizeTemp.shape[::-1]
    cannyTemp = cv2.Canny(resizeTemp, 40, 140, L2gradient=True)
    # cv2.imshow("resized image", cannyTemp)
    # cv2.waitKey(0)

    matchedRes = cv2.matchTemplate(cannyTemp, cannyImg, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchedRes)
    print(max_val, max_loc)

    img2 = resizedImg.copy()
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img2, top_left, bottom_right, 255, 4)

    plt.subplot(121), plt.imshow(matchedRes, cmap='gray')
    plt.title('Heat Map'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(f'Template scaled to {scale:.2f}, iteration = {index+1}')
    plt.show()




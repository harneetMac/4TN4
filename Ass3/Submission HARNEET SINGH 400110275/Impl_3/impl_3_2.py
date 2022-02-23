import numpy as np
import cv2
from matplotlib import pyplot as plt

# reading the template as gray scale
template = cv2.imread('circle.bmp', cv2.IMREAD_GRAYSCALE)
template = cv2.Canny(template, 40, 90, L2gradient=True)
w, h = template.shape[::-1]

# reading the given image and resize it to a factor of 2
# this resized image will be used to find the circle
img = cv2.imread('messi.jpg', cv2.IMREAD_GRAYSCALE)
resizedImg = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

img2 = img.copy()
img = resizedImg
width, height = img.shape[::-1]

# 0.1 scaling level is used, e.g. going from 100% to 90% to 80% and so on...
percentSpacing = 0.1
scalingFactor = np.arange(1, 0, -percentSpacing)

# looping over all the scaling factors to match the template
# result is shown as we apply the matchTemplate()
for index, scale in enumerate(scalingFactor):
    resizeImg = cv2.resize(resizedImg, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cannyImg = cv2.Canny(resizeImg, 40, 140, L2gradient=True)
    # cv2.imshow("resized image", cannyImg)
    # cv2.waitKey(0)

    matchedRes = cv2.matchTemplate(cannyImg, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchedRes)
    print(max_val, max_loc)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(resizeImg, top_left, bottom_right, 255, 4)

    plt.subplot(121), plt.imshow(matchedRes, cmap='gray')
    plt.title('Heat Map'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(resizeImg, cmap='gray', vmin=0, vmax=255)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(f'Scaled down to {scale:.2f}, iteration = {index+1}')
    plt.show()




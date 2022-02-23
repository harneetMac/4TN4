import numpy as np
import cv2
from matplotlib import pyplot as plt

# reading the image and creating a copy to place a
# rectangle on the circle
img = cv2.imread('messi.jpg', cv2.IMREAD_GRAYSCALE)
cannyImg = cv2.Canny(img, 50, 150, L2gradient=True)
img2 = img.copy()
img = cannyImg

template = cv2.imread('circle.bmp', cv2.IMREAD_GRAYSCALE)
template = cv2.Canny(template, 50, 150, L2gradient=True)
w, h = template.shape[::-1]

# using matchTemplate() to template match
matchedRes = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchedRes)
print(max_val, max_loc)

# these two variables represent the corners of the rectangle
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img2, top_left, bottom_right, 255, 4)

# ploting the heat map with image
plt.subplot(211), plt.imshow(matchedRes, cmap = 'gray')
plt.title('Heat Map'), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(img2, cmap = 'gray', vmin=0, vmax=255)
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()

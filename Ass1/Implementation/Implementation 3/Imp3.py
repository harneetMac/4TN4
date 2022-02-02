import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading the image as is and storing it in a variable
img = cv2.imread('selfie1.jpg', cv2.IMREAD_COLOR)
# print(img.shape)

#converting the color space to HSV from BGR
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#calculating and plotting the histogram to discover the hue range
hist_hue = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])
# print(hist_hue)
plt.plot(hist_hue, color='r', label = 'hue')
plt.xlim([0, 180])
plt.legend()
plt.show()

# define range in HSV
lower_range = np.array([0,30,30])
upper_range = np.array([20,255,255])
# Threshold the HSV image to get only skin color
mask = cv2.inRange(hsv_img, lower_range, upper_range)
# Bitwise-AND mask and original image
result = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow('HUE', result)

# if image processed correctly, it will be saved by pressing 's',
#otherwise quit the window
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("impl3.png", result)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

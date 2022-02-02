import cv2
import numpy as np
from matplotlib import pyplot as plt

kernelSize = 3

img3 = cv2.imread('img3.png')
medianImg3 = cv2.medianBlur(img3, kernelSize)

cv2.imshow("MedianFilteredImg3", medianImg3)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("impl3Median.jpg", medianImg3)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

import cv2
import numpy as np
from matplotlib import pyplot as plt

kernelSize = 0
sigmaX = 1
sigmaY = 1

img3 = cv2.imread('img3.png')
gaussianImg3 = cv2.GaussianBlur(img3, (kernelSize, kernelSize), sigmaX, sigmaY)

cv2.imshow("GaussianFilteredImg3", gaussianImg3)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite(f"impl3_2Gaussian_k={kernelSize}_sigmaX={sigmaX}.jpg", gaussianImg3)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

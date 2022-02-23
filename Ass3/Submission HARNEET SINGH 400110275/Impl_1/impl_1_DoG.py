import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('lp.jpg')
# attempt to try it in gray scale

# setting kernel/sigma values
kernelSizeGaussian_1 = 3       # 17
kernelSizeGaussian_2 = 1       # 15
# letting function compute sigma values
sigmaX_1 = 0
sigmaX_2 = 0

# calling gaussian filter twice to set up DoG
gaussianFiltered_1 = cv2.GaussianBlur(img, (kernelSizeGaussian_1, kernelSizeGaussian_1), sigmaX_1)
gaussianFiltered_2 = cv2.GaussianBlur(img, (kernelSizeGaussian_2, kernelSizeGaussian_2), sigmaX_2)
# cv2.imshow("low standard deviation", gaussianFiltered_1)
# cv2.imshow("large standard deviation", gaussianFiltered_2)
# cv2.waitKey(0)

# DoG function: subtracting one from another
DoGImg = (cv2.subtract(gaussianFiltered_1, gaussianFiltered_2))
cv2.imshow("DoG", DoGImg)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('Impl_1.jpg', DoGImg)

cv2.destroyAllWindows()

import numpy as np
import cv2

img = cv2.imread('lp.jpg')

# preprocessing image using gaussian filter
kernelSizeGaussian = 9
gaussianFiltered = cv2.GaussianBlur(img, (kernelSizeGaussian, kernelSizeGaussian),
                                    0, borderType=cv2.BORDER_CONSTANT)

# based on sigma value and median of the image, low and high threshold will be
# calculated. The idea is to pick values that are close to the median.
sigma = 3
median = np.median(img)
lowThreshold = int(max(0, (1.0 - sigma) * median))
highThreshold = int(min(255, (1.0 + sigma) * median))
print(f'low threshold = {lowThreshold}, high threshold = {highThreshold}')

# calling Canny() function to get the license plate only
cannyOperatedImg = cv2.Canny(gaussianFiltered, lowThreshold, highThreshold,
                            None, L2gradient=True)

cv2.imshow("Canny Operated Image", cannyOperatedImg)
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite('Impl_1_1.jpg', cannyOperatedImg)

cv2.destroyAllWindows()

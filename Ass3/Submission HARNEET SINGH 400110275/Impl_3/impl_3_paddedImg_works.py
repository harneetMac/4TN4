import numpy as np
import cv2
from matplotlib import pyplot as plt

# reading template and applying Canny operator to remove noise
template = cv2.imread('circle.bmp', cv2.IMREAD_GRAYSCALE)
template = cv2.Canny(template, 50, 150, L2gradient=True)
w, h = template.shape[::-1]
# computing the cross correlation terms, here computing for
# template window
meanTemplate = np.mean(template)
templMinusMean = np.subtract(template, meanTemplate)
templMinusMeanSquare = np.square(templMinusMean)
templMinusMeanSqSum = np.sum(templMinusMeanSquare)

# reading image and applying Canny Operator to remove noise
# creating padding around the image to go through all the pixels
# in the image
img = cv2.imread('messi.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.copyMakeBorder(img, w//2, w//2, h//2, h//2, cv2.BORDER_REFLECT)
cannyImg = cv2.Canny(img, 47, 75, L2gradient=True)
# cv2.imshow("Canny Image", cannyImg)
# cv2.waitKey(0)
img2 = img.copy()
img = cannyImg
width, height = img.shape[::-1]

matchedRes = np.zeros((height-h+1, width-w+1))

# convoluting with the image by passin the window at every pixel
# in the for loop, calculating cross correlation terms because
# it will change for the image each time
for k in range(height - h + 1):
    for p in range(width - w + 1):
        subMat = img[k:k+h, p:p+w]
        meanSubMat = np.mean(subMat)
        subMatMinusMean = np.subtract(subMat, meanSubMat)
        subMatMinusMeanSq = np.square(subMatMinusMean)
        subMatMinusMeanSqSum = np.sum(subMatMinusMeanSq)

        numMultiply = np.multiply(templMinusMean, subMatMinusMean)
        numerator = np.sum(numMultiply)

        denMultiply = templMinusMeanSqSum * subMatMinusMeanSqSum
        denominator = (denMultiply) ** (1/2)

        if denominator == 0:
            print(f'Zero Division at {k}, {p}')
            continue
        else:
            matchedRes[k, p] = numerator/denominator

# saving the heat map in the image shape array
result = np.zeros((img.shape))
result[h//2:(h//2)+height-h+1, w//2:(w//2)+width-w+1] = matchedRes

# finding min and max values and location
# but will require max value and location in this case
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(f'max value: {max_val} at {max_loc}')

# setting corner values for the rectange that will be drawn on the
# image
top_left = max_loc[0] - (w//2), max_loc[1] - (h//2)
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img2, top_left, bottom_right, 255, 4)

# plotting the image with its heat map
plt.subplot(211), plt.imshow(result, cmap = 'gray')
plt.title('Heat Map'), plt.xticks([]), plt.yticks([])
plt.subplot(212), plt.imshow(img2, cmap = 'gray', vmin=0, vmax=255)
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img4 = cv2.imread('img4.png')
cv2.imshow("original", img4)
labImg4 = cv2.cvtColor(img4, cv2.COLOR_BGR2LAB)

l, a, b = cv2.split(labImg4)

lEqualized = cv2.equalizeHist(l)

mergedLabImg = cv2.merge((lEqualized, a, b))
equalizedBgrImg = cv2.cvtColor(mergedLabImg, cv2.COLOR_LAB2BGR)
cv2.imshow("L-Equalized Image", equalizedBgrImg)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("Impl4_2_L-Equalized.png", equalizedBgrImg)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

plt.subplot(121), plt.hist(l.ravel(), 256)
plt.title(f'L - Hist'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.hist(lEqualized.ravel(), 256)
plt.title(f'L - Equalized Hist'), plt.xticks([]), plt.yticks([])
plt.show()



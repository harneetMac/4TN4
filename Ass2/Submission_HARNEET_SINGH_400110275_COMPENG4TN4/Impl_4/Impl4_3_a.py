import cv2
import numpy as np
from matplotlib import pyplot as plt

img4 = cv2.imread('img4.png')
# cv2.imshow("original", img4)

b, g, r = cv2.split(img4)

claheObject = cv2.createCLAHE(clipLimit = 3.0, tileGridSize = (8,8))
bEqualized = claheObject.apply(b)
gEqualized = claheObject.apply(g)
rEqualized = claheObject.apply(r)

mergedImg = cv2.merge((bEqualized, gEqualized, rEqualized))
cv2.imshow("Equalized Image", mergedImg)

if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("Impl4_3_a_Equalized.png", mergedImg)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

plt.subplot(321), plt.hist(b.ravel(), 256, color='blue')
plt.title(f'Blue Hist'), plt.xticks([]), plt.yticks([])
plt.subplot(323), plt.hist(g.ravel(), 256, color='green')
plt.title(f'Green Hist'), plt.xticks([]), plt.yticks([])
plt.subplot(325), plt.hist(r.ravel(), 256, color='red')
plt.title(f'Red Hist'), plt.xticks([]), plt.yticks([])

plt.subplot(322), plt.hist(bEqualized.ravel(), 256, color='blue')
plt.title(f'Blue Equalized'), plt.xticks([]), plt.yticks([])
plt.subplot(324), plt.hist(gEqualized.ravel(), 256, color='green')
plt.title(f'Green Equalized'), plt.xticks([]), plt.yticks([])
plt.subplot(326), plt.hist(rEqualized.ravel(), 256, color='red')
plt.title(f'Red Equalized'), plt.xticks([]), plt.yticks([])
plt.show()


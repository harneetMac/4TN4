import cv2
import numpy as np

img = np.zeros((6,6), dtype = 'uint8')
givenimg = np.array([[3,5,8,4],
                     [9,1,2,9],
                     [4,6,7,3],
                     [3,8,5,4]])
img[1:5, 1:5] = givenimg
img = np.asmatrix(img)
kernel = np.ones((3,3), dtype = 'uint8')
shapeKernel = int(kernel.shape[0])
filteredImg = np.empty_like(givenimg, dtype = 'uint8')

def medianFilter(n):
    for k in range(n+1):
        for p in range(n+1):
            subMatrix = np.ravel(img[k:k+n, p:p+n])
            # print(subMatrix)
            medianValue = (np.median(subMatrix)).astype('uint8')
            # print(medianValue)
            filteredImg[k, p] = medianValue
    print(filteredImg)

medianFilter(shapeKernel)


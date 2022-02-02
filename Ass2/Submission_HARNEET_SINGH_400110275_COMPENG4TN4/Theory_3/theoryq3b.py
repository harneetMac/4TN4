import cv2
import numpy as np

kernel = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]])
shapeKernel = int(kernel.shape[0])
img = np.zeros((6,6), dtype = 'uint8')
givenimg = np.array([[3,5,8,4],
                     [9,1,2,9],
                     [4,6,7,3],
                     [3,8,5,4]])
img[1:5, 1:5] = givenimg
img = np.asmatrix(img)
filteredImg = np.empty_like(givenimg, dtype = 'uint8')

def medianFilter(n):
    for k in range(n+1):
        for p in range(n+1):
            elements = []
            subMatrix = np.ravel(img[k:k+n, p:p+n])
            elements.append(subMatrix[1])
            elements.append(subMatrix[3])
            elements.append(subMatrix[4])
            elements.append(subMatrix[5])
            elements.append(subMatrix[7])

            medianValue = (np.median(elements)).astype('uint8')
            filteredImg[k, p] = medianValue
    print(filteredImg)

medianFilter(shapeKernel)


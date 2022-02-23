import numpy as np
import cv2

# creating image array
img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='uint8')
imgShape = int(img.shape[1])
dilateImg = np.zeros_like(img)

# creating structuring element
structElement = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype='uint8')
structElemShape = int(structElement.shape[1])

# finding elements that need to be considered based on structuring element
elem = np.where(structElement==1)
elements = list(zip(elem[0], elem[1]))
# print(elements)

# dilate functionality implemented in the for loop
# first getting the sub-matrix, then using the elements variable,
# determining the elements of interest based on structuring element
for k in range(imgShape-structElemShape+1):
    for p in range(imgShape-structElemShape+1):
        subMatrix = img[k:k+structElemShape, p:p+structElemShape]

        temp = 0
        for loc in elements:
            temp = temp or subMatrix[loc[0]][loc[1]]

        if temp:
            dilateImg[k+1][p+1] = 1

print(dilateImg)

# cv2's dilate function to check the functionality of my logic
dilateImg2 = cv2.dilate(img, structElement, iterations=1)
print(dilateImg2)


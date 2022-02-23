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
erodedImg = np.zeros_like(img)

# creating structuring element
structElement = np.array([[0, 1, 0],
                          [0, 1, 1],
                          [0, 1, 0]], dtype='uint8')
structElemShape = int(structElement.shape[1])

# finding elements that need to be considered based on structuring element
elem = np.where(structElement==1)
elements = list(zip(elem[0], elem[1]))
# print(elements)

# erode functionality implemented in the for loop
for k in range(imgShape-structElemShape+1):
    for p in range(imgShape-structElemShape+1):
        subMatrix = img[k:k+structElemShape, p:p+structElemShape]

        temp = 1
        for loc in elements:
            temp = temp and subMatrix[loc[0]][loc[1]]

        if temp:
            erodedImg[k+1][p+1] = 1

print(erodedImg)

# cv2's erode function to check the functionality of my logic
erodedImg2 = cv2.erode(img, structElement, iterations=1)
print(erodedImg2)


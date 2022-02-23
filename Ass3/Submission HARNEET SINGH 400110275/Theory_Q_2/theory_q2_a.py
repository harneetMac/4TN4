import numpy as np

# creating the given image
img = np.zeros((8,8))
givenImg = np.array([[127, 127, 127, 0, 0, 0],
                     [0, 127, 127, 127, 0, 0],
                     [0, 0, 80, 80, 80, 0],
                     [0, 0, 0, 127, 127, 127],
                     [0, 0, 0, 0, 127, 127],
                     [0, 0, 0, 0, 0, 0]])
img[1:7, 1:7] = givenImg
imgShape = int(img.shape[0])

# creating Sobal operators
kernelSobelX = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
kernelSobelY = np.copy(kernelSobelX)
kernelSobelY = np.transpose(kernelSobelY)
kernelShape = int(kernelSobelX.shape[0])

# applying Sobel filter
def sobelFilter(imgSize, n, kernel):
    filteredImg = np.zeros_like(img)

    for k in range(imgSize - n + 1):
        for p in range(imgSize - n + 1):
            subMatrix = np.zeros((n, n))
            subMatrix = img[k:k + n, p:p + n]
            multipliedValue = np.multiply(subMatrix, kernel)
            sumAllElements = np.sum(multipliedValue)

            filteredImg[k + 1, p + 1] = sumAllElements
    return filteredImg

# applying magnitude and phase of the gradient
def gradientMagPhase(gradX, gradY):
    gradXSquare = np.square(gradX)
    gradYSquare = np.square(gradY)
    gradxAddGradY = np.add(gradXSquare, gradYSquare)

    gMagnitude = np.sqrt(gradxAddGradY)
    gPhase = np.arctan2(gradY, gradX) * 180 / np.pi

    return gMagnitude, gPhase


# Computing the gradient in x and y directions
Gx = sobelFilter(imgShape, kernelShape, kernelSobelX)
Gy = sobelFilter(imgShape, kernelShape, kernelSobelY)
# print(f"Gx =\n {Gx}")
# print(f"Gy =\n {Gy}")

# Computing the gradient magnitude and phase in degree
gradMag, gradPhase = gradientMagPhase(Gx, Gy)
print(f"Mag =\n {gradMag}")
# print(f"Phase =\n {gradPhase}")

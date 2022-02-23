import numpy as np
import cv2

lowThreshold = 20
highThreshold = 120

img = np.zeros((8, 8))
givenImg = np.array([[127, 127, 127, 0, 0, 0],
                     [0, 127, 127, 127, 0, 0],
                     [0, 0, 80, 80, 80, 0],
                     [0, 0, 0, 127, 127, 127],
                     [0, 0, 0, 0, 127, 127],
                     [0, 0, 0, 0, 0, 0]])
img[1:7, 1:7] = givenImg
# img = img.astype('uint8')
imgShape = int(img.shape[0])

kernelSobelX = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
kernelSobelY = np.copy(kernelSobelX)
kernelSobelY = np.transpose(kernelSobelY)
kernelShape = int(kernelSobelX.shape[0])


def gaussianFilter(img, kernelSize):
    blurredImg = cv2.GaussianBlur(img, (kernelSize, kernelSize), 0)
    return blurredImg


def sobelFilter(image, imgSize, n, kernel):
    filteredImg = np.zeros_like(image)

    for k in range(imgSize - n + 1):
        for p in range(imgSize - n + 1):
            subMatrix = np.zeros((n, n))
            subMatrix = image[k:k + n, p:p + n]
            multipliedValue = np.multiply(subMatrix, kernel)
            sumAllElements = np.sum(multipliedValue)

            filteredImg[k + 1, p + 1] = sumAllElements
    return filteredImg


def gradientMagPhase(gradX, gradY):
    gradXSquare = np.square(gradX)
    gradYSquare = np.square(gradY)
    gradxAddGradY = np.add(gradXSquare, gradYSquare)

    gMagnitude = np.sqrt(gradxAddGradY)
    gPhase = np.arctan2(gradY, gradX) * 180 / np.pi

    return gMagnitude, gPhase


def nonMaximumSuppression(mag, phase, imgSize, kernelSize):
    suppressedImg = np.copy(mag)
    maxGradValue = np.amax(mag)

    for k in range(imgSize - kernelSize + 1):
        for p in range(imgSize - kernelSize + 1):
            tempPhase = phase[k+1][p+1]

            if (-22.5 <= tempPhase < 22.5) or (157.5 <= tempPhase < 180) or (-180 <= tempPhase < -157.5):
                middle = mag[k + 1][p + 1]
                topCenter = mag[k][p + 1]
                bottomCenter = mag[k + 2][p + 1]
                if middle < topCenter or middle < bottomCenter:
                    suppressedImg[k + 1][p + 1] = 0

            elif (22.5 <= tempPhase < 67.5) or (-157.5 <= tempPhase < -112.5):
                middle = mag[k + 1][p + 1]
                topLeft = mag[k][p]
                bottomRight = mag[k + 2][p + 2]
                if middle < topLeft or middle < bottomRight:
                    suppressedImg[k + 1][p + 1] = 0

            elif (67.5 <= tempPhase < 112.5) or (-112.5 <= tempPhase < -67.5):
                middle = mag[k + 1][p + 1]
                middleLeft = mag[k + 1][p]
                middleRight = mag[k + 1][p + 2]
                if middle < middleLeft or middle < middleRight:
                    suppressedImg[k + 1][p + 1] = 0

            elif (112.5 <= tempPhase < 157.5) or (-67.5 <= tempPhase < -22.5):
                middle = mag[k + 1][p + 1]
                topRight = mag[k][p + 2]
                bottomLeft = mag[k + 2][p]
                if middle < topRight or middle < bottomLeft:
                    suppressedImg[k + 1][p + 1] = 0

            else:
                raise Exception("Sorry, could not classify")

    # binding the values to range of 0 to 255
    suppressedImg = (suppressedImg / maxGradValue) * 255
    return np.round(suppressedImg)


def threshold(image, lowThresh, highThresh):

    threshResult = np.zeros_like(image, dtype=np.int32)

    weak = np.uint8(lowThresh)
    strong = np.uint8(255)

    strong_i, strong_j = np.where(image >= highThresh)
    zeros_i, zeros_j = np.where(image < lowThresh)
    weak_i, weak_j = np.where((image <= highThresh) & (image >= lowThresh))

    threshResult[strong_i, strong_j] = strong
    threshResult[weak_i, weak_j] = weak
    threshResult[zeros_i, zeros_j] = 0

    return threshResult


def recursiveOperationHysteresis(image, i, j):
    weak = 20

    if image[i + 1, j - 1] == weak:
        p = i + 1
        q = j - 1

    elif image[i + 1, j] == weak:
        p = i + 1
        q = j

    elif image[i + 1, j + 1] == weak:
        p = i + 1
        q = j + 1

    elif image[i, j - 1] == weak:
        p = i
        q = j - 1

    elif image[i, j + 1] == weak:
        p = i
        q = j + 1

    elif image[i - 1, j - 1] == weak:
        p = i - 1
        q = j - 1

    elif image[i - 1, j] == weak:
        p = i - 1
        q = j

    elif image[i - 1, j + 1] == weak:
        p = i - 1
        q = j + 1

        image[i, j] = weak
    else:
        return None

    if p != i or q != j:
        image[p, q] = 1
        recursiveOperationHysteresis(image, p, q)


def hysteresis(image, weak):
    maxValue = 255
    edge = 1
    hysteresisRes = np.copy(image)

    for i in range(imgShape-kernelShape+1):
        for j in range(imgShape-kernelShape+1):
            if image[i, j] == weak:
                if recursiveOperationHysteresis(image, i, j) is None:
                    image[i, j] = 0

    hysteresisRes = np.where(image == maxValue, edge, 0)
    return hysteresisRes


# computing the gaussian blur on the image to smoothen it
blurredImg = gaussianFilter(img, kernelShape)

# Computing the gradient in x and y directions
Gx = sobelFilter(blurredImg, imgShape, kernelShape, kernelSobelX)
Gy = sobelFilter(blurredImg, imgShape, kernelShape, kernelSobelY)

# Computing the gradient magnitude and phase in degree
gradMag, gradPhase = gradientMagPhase(Gx, Gy)
# print(gradMag)
# print(gradPhase)

nmsImg = nonMaximumSuppression(gradMag, gradPhase, imgShape, kernelShape)
thresholdResult = threshold(nmsImg, lowThreshold, highThreshold)
hysteresisResult = hysteresis(thresholdResult, lowThreshold)
print(hysteresisResult)


# output using Canny Edge Detection of OpenCV
img2 = np.zeros_like(img)
edgeDetected = cv2.Canny(img.astype('uint8'), lowThreshold, highThreshold, img2, kernelShape, True)

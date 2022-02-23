import numpy as np
import cv2

img = cv2.imread('messi.jpg', cv2.IMREAD_GRAYSCALE)
paddedImg = cv2.copyMakeBorder(img, 83, 83, 87, 87, cv2.BORDER_REPLICATE)
imgHeight, imgWidth = img.shape
PadimgHeight, PadimgWidth = paddedImg.shape


template = cv2.imread('circle.bmp', cv2.IMREAD_GRAYSCALE)
templateHeight, templateWidth = template.shape
template = template[:,:templateWidth-1]
templateHeight, templateWidth = template.shape
avgTemplate = np.mean(template)

cannyOperatedImg = cv2.Canny(paddedImg, 25, 80, None, L2gradient=True)
cannyOperatedTemp = cv2.Canny(template, 50, 150)
# print(cannyOperatedImg.shape)
# cv2.imshow("Canny Operated Image", cannyOperatedTemp)
# cv2.waitKey(0)


def crossCorrelation(imgH, imgW, templateH, templateW, templ):
    normalCorrImg = np.zeros((imgH, imgW))

    subtTempl = np.subtract(templ, avgTemplate)
    subtTemplSquared = np.sum(np.square(subtTempl))

    for k in range(PadimgHeight-templateH+1):
        for p in range(PadimgWidth-templateW+1):
            print(k,p)
            subMatrix = np.zeros((templateH, templateW))
            subMatrix = cannyOperatedImg[k+83:k+83+templateH, p+87:p+87+templateW]
            # print(k, p, subMatrix.shape)

            avgSubMatrix = np.mean(subMatrix)

            subtSubMatrix = np.subtract(subMatrix, avgSubMatrix)
            subtSubMatrixSquared = np.sum(np.square(subtSubMatrix))

            numerator = np.sum(np.multiply(subtTempl, subtSubMatrix))
            denominator = (subtTemplSquared * subtSubMatrixSquared)**(0.5)

            normalCorrImg[k, p] = numerator/denominator

    return normalCorrImg


crossCorrelated = crossCorrelation(imgHeight, imgWidth, templateHeight, templateWidth, template)
print(np.amax(crossCorrelated))
print(np.amin(crossCorrelated))
heat_map = crossCorrelated[crossCorrelated > 0.1]
print(heat_map)

location = np.where(crossCorrelated > 0.1)
print(location)

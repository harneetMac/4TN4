import cv2
import numpy as np
import math

# creating the image
img = np.array([[0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 0, 0, 0, 0]])
imgShape = int(img.shape[0])
resultImg = np.zeros_like(img)

# creating kernels for x and y directions
kernelX = np.array([[-1, 1]])
kernelY = np.copy(kernelX)
kernelY = np.transpose(kernelY)
kernelX_h, kernelX_w = kernelX.shape
kernelY_h, kernelY_w = kernelY.shape


# function to extract gradients using the kernel (convolving kernel over the image)
def gradExtraction(kernel, ker_h, ker_w):
    filteredImg = np.zeros_like(img)

    for k in range(imgShape - ker_h + 1):
        for p in range(imgShape - ker_w + 1):
            subMatrix = np.zeros((ker_h, ker_w))
            subMatrix = img[k:k + ker_h, p:p + ker_w]
            multipliedValue = np.multiply(subMatrix, kernel)
            sumAllElements = np.sum(multipliedValue)

            filteredImg[k, p] = sumAllElements
    return filteredImg


# used to find max and min lambda values
def findMinMax(num1, num2):
    if num1 > num2:
        return num1, num2
    else:
        return num2, num1


# following variables used to calculate M, R and eigen values
I_x = gradExtraction(kernelX, kernelX_h, kernelX_w)
I_y = gradExtraction(kernelY, kernelY_h, kernelY_w)
# print(f'{I_x=}')
# print(f'{I_y=}')

I_xSquared = np.square(I_x)
I_ySquared = np.square(I_y)
# print(f'{I_xSquared=}')
# print(f'{I_ySquared=}')

Ix_Iy = np.multiply(I_x, I_y)

M = np.zeros((9, 2, 2))
i = 0
alpha = 0.04

# going through each non-zero pixel at a time to get lambda max and min, and cornerness score
for h in range(1, 4):
    for w in range(1, 4):
        sum_I_xSquared = int(np.sum(I_xSquared[h-1:h+1, w-1:w+1]))
        sum_I_ySquared = int(np.sum(I_ySquared[h-1:h+1, w-1:w+1]))
        sum_Ix_Iy = int(np.sum(Ix_Iy[h-1:h+1, w-1:w+1]))
        M[i] = [[sum_I_xSquared,    sum_Ix_Iy],
                [sum_Ix_Iy,         sum_I_ySquared]]
        print(f'{i}{M[i]=}')

        sqrtTerm = 4*sum_Ix_Iy*sum_Ix_Iy + ((sum_I_xSquared-sum_I_ySquared)**2)
        lambda_1 = (1/2) * ((sum_I_xSquared+sum_I_ySquared) + math.sqrt(sqrtTerm))
        lambda_2 = (1/2) * ((sum_I_xSquared+sum_I_ySquared) - math.sqrt(sqrtTerm))

        lambdaMax, lambdaMin = findMinMax(lambda_1, lambda_2)
        # print(f'{i}: {lambdaMin=},\t\t {lambdaMax=}')

        # detect corners if above certain threshold:
        # cornerness score:
        R = lambda_1 * lambda_2 - alpha * (lambda_1 + lambda_2) ** 2
        # print(f'{R=}')

        # generally, we can apply a normalization function to the M matrix
        # in this case, since we need true lambda values, I've omitted the normalization function
        # threshold is computed based on the trend between all non-zero values
        # higher value could have been chosen but a value of 1000 works as well
        # (essentially a large positive value will be a good candidate for threshold)
        # Note that I am assuming top-left corner is (0 , 0) in the image
        if R > 1000:
            # print(f'M{i}: \n{M[i]} with {R=}, {lambdaMax=} and {lambdaMin=}. Corner at location ({h}, {w})\n')
            resultImg[h, w] = 255

        # for comparison with my lambda values:
        eigVal, eigVec = np.linalg.eig(M[i])
        # print(f'{i}: {eigVal}, \n{eigVec}')

        i = i + 1

print(f'Result Image with corners is \n{resultImg}')

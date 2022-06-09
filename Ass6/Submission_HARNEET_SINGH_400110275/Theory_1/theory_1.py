import cv2
import numpy as np
import matplotlib.pyplot as plt

# read given images
img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
                [0, 255, 0, 255, 0, 0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 255, 0, 0, 255, 0],
                [0, 0, 0, 0, 0, 255, 0, 0, 255, 0],
                [0, 0, 0, 0, 0, 255, 0, 0, 255, 0],
                [0, 0, 0, 0, 0, 255, 255, 255, 255, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
img_height, img_width = img.shape[:2]

# diagonal distance between the image (max rho value)
diagonal = np.sqrt(np.square(img_height) + np.square(img_width))

# design accumulator based on number of theta (180) and number of rhos (180) we need
tick_rho = 2 * diagonal/180
tick_theta = 1

rhos = np.arange(-diagonal, diagonal, tick_rho)
thetas = np.arange(0, 180, tick_theta)
# print(rhos)
# print(thetas)

cosTheta = np.cos(np.deg2rad(thetas))
sinTheta = np.sin(np.deg2rad(thetas))

# defining an accumulator of size 180x180
accumulator = np.zeros((len(rhos), len(thetas)))
# print(accumulator.shape)

for i in range(img_height):
    for j in range(img_width):
        if img[i, j] != 0:
            # image origin is in the middle
            # print(f'image index is {i - img_height / 2}, {j - img_width / 2}')

            # need lists to store rho and theta values
            rhoList, thetaList = [], []

            for theta in thetas:
                rho = ((j - img_width / 2) * (cosTheta[theta])) + ((i - img_height / 2) * (sinTheta[theta]))
                # print(rho, theta)
                rhoList.append(rho)
                thetaList.append(theta)

                # now we need to find the index of row where we need to increment the accumulator

                rhoIndex = np.argmin(np.abs(rhos - rho))
                # print(f'{rhoIndex}')

                accumulator[rhoIndex, theta] += 1

accumulatorMax = np.amax(accumulator)
print(f'max value in accumulator is: {accumulatorMax}')

# now we need to iterate through the accumulator matrix to find the max value
numMaxValues = 0
for k in range(len(rhos)):
    for l in range(len(thetas)):
        if accumulator[k, l] == accumulatorMax:
            rho = round(rhos[k])
            theta = thetas[l]
            print(f'{k}, {l}, rhos is {rho}, and theta is {theta}')
            numMaxValues += 1

print(f'number of max values in the accumulator are {numMaxValues}')



import cv2
import numpy as np
import matplotlib.pyplot as plt

# read given images
img0 = np.array([[0, 0, 0, 0, 0, 0],
                 [0, 255, 255, 255, 0, 0],
                 [0, 255, 0, 255, 0, 0],
                 [0, 0, 0, 0, 0, 0]])

img1 = np.array([[0, 0, 0, 0, 0, 0],
                 [0, 0, 255, 255, 255, 0],
                 [0, 0, 255, 0, 255, 0],
                 [0, 0, 0, 0, 0, 0]])
imgHeight, imgWidth = img0.shape[:2]

# zero padding
frame0 = np.zeros((imgHeight+2, imgWidth+2), dtype='uint8')
frame1 = np.zeros((imgHeight+2, imgWidth+2), dtype='uint8')
frame0[1:5, 1:7] = img0
frame1[1:5, 1:7] = img1

Grad_Ix = np.zeros_like(frame0, dtype='int')
Grad_Iy = np.zeros_like(frame0, dtype='int')
Grad_It = np.zeros_like(frame0, dtype='int')

# find partial derivative in x-direction
for h in range(1, imgHeight+1):
    for k in range(1, imgWidth+1):
        # I_x = (1/4) * (frame1[h, k] + frame1[h+1, k] + frame0[h, k] + frame0[h+1, k])
        # I_xPlus1 = (1/4) * (frame1[frame1[h, k+1] + frame1[h+1, k+1] + frame0[h, k+1] + frame0[h+1, k+1]])
        x_frame0 = np.sum(frame0[h:h+2, k:k+1])
        x_frame1 = np.sum(frame1[h:h+2, k:k+1])

        xPlus1_frame0 = np.sum(frame0[h:h+2, k+1:k+2])
        xPlus1_frame1 = np.sum(frame1[h:h+2, k+1:k+2])

        xPlus1 = (xPlus1_frame0 + xPlus1_frame1) * (1/4)
        xPlus0 = (x_frame0 + x_frame1) * (1/4)

        Grad_Ix[h+1, k+1] = xPlus1 - xPlus0


# find partial derivative in y-direction
for h in range(1, imgHeight+1):
    for k in range(1, imgWidth+1):
        y_frame0 = np.sum(frame0[h:h+1, k:k+2])
        y_frame1 = np.sum(frame1[h:h+1, k:k+2])

        yPlus1_frame0 = np.sum(frame0[h+1:h+2, k:k+2])
        yPlus1_frame1 = np.sum(frame1[h+1:h+2, k:k+2])

        yPlus1 = (yPlus1_frame0 + yPlus1_frame1) * (1/4)
        yPlus0 = (y_frame0 + y_frame1) * (1/4)

        Grad_Iy[h+1, k+1] = yPlus1 - yPlus0

# find partial derivative in t-direction
for h in range(1, imgHeight+1):
    for k in range(1, imgWidth+1):
        t_frame0 = np.sum(frame0[h:h+2, k:k+2], dtype='int')
        t_frame1 = np.sum(frame1[h:h+2, k:k+2], dtype='int')

        t = (t_frame1 - t_frame0) * (1/4)

        Grad_It[h+1, k+1] = t

Ix = Grad_Ix[1:5, 1:7]
Iy = Grad_Iy[1:5, 1:7]
It = Grad_It[1:5, 1:7]

print(f'Ix is: \n {Ix}')
print(f'Iy is: \n {Iy}')
print(f'It is: \n {It}')

# Ix_square = np.square(Ix)
# Iy_square = np.square(Iy)
# I_xy = np.multiply(Ix, Iy)
# I_xt = np.multiply(Ix, It)
# I_yt = np.multiply(Iy, It)

# A_transpose_A = np.array((2, 2), dtype='int')
# A_transpose_b = np.array((2, 1), dtype='int')
# # applying least square method to find u-vector for each pixel
# for h in range(0, imgHeight):
#     for k in range(0, imgWidth):
#         A_transpose_A = [[Ix_square[h, k],  I_xy[h, k]],
#                          [I_xy[h, k],       Iy_square[h, k]]]
#
#         A_transpose_b = [[-I_xt],
#                          [-I_yt]]
#
#         u = np.linalg.lstsq(A_transpose_A, A_transpose_b, rcond=None)

A = np.zeros((9, 2), dtype='int')
b = np.zeros((9, 1), dtype='int')

index = 0
# applying least square method to find u-vector for each pixel
for h in range(0, imgHeight):
    for k in range(0, imgWidth):
        i_iterator = 0

        subMatrixIx = Grad_Ix[h:h+3, k:k+3]
        subMatrixIy = Grad_Iy[h:h+3, k:k+3]
        subMatrixIt = Grad_It[h:h+3, k:k+3]

        for i in range(3):
            for j in range(3):
                A[i_iterator] = [subMatrixIx[i, j], subMatrixIy[i, j]]
                b[i_iterator] = -subMatrixIt[i, j]
                i_iterator += 1

        u = np.linalg.lstsq(A, b, rcond=None)[0]

        print(f'At index: {index} ({h}, {k}): u is:\n{u}', end='\n\n')
        index += 1


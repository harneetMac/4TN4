import cv2
import numpy as np
from matplotlib import pyplot as plt

messiImg = cv2.imread('messi.jpg', cv2.IMREAD_GRAYSCALE)
ronaldoImg = cv2.imread('ronaldo.jpg', cv2.IMREAD_GRAYSCALE)

# Creating a circular LPF for messi's image
def createLPF(img):
    rows, cols = img.shape
    centrePointRow, centrePointcol = int(rows / 2), int(cols / 2)
    center = [centrePointRow, centrePointcol]
    # below, third argument (2) is there to match DFT conversion
    lpfMask = np.zeros((rows, cols, 2), np.uint8)
    radius = 25
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    lpfMask[mask_area] = 1
    return lpfMask

def fft2ifft(img, mask):
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    # img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back

def plotImg(filteredImg, img=None):
    fig = plt.figure(figsize=(1,1))
    if img is not None:
        plt.subplot(122), plt.imshow(img, cmap = 'gray', vmin=0, vmax=255)
        plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(121), plt.imshow(filteredImg, cmap = 'gray', vmin=0, vmax=255)
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        plt.subplot(111), plt.imshow(filteredImg, cmap = 'gray', vmin=0, vmax=255)
        plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
        plt.show()

def addimages(img1, img2):
    halfImg1 = np.multiply(img1, 0.5)
    halfImg2 = np.multiply(img2, 0.5)
    return halfImg1 + halfImg2

lpfMask = createLPF(messiImg)
hpfMask = 1 - lpfMask

filteredMessiImg = fft2ifft(messiImg, lpfMask)
cv2.imwrite("messi + LPF.jpg", filteredMessiImg)
plotImg(messiImg, filteredMessiImg)
filteredRonaldoImg = fft2ifft(ronaldoImg, hpfMask)
cv2.imwrite("ronaldo + HPF.jpg", filteredRonaldoImg)
plotImg(ronaldoImg, filteredRonaldoImg)

addedImages = addimages(filteredMessiImg, filteredRonaldoImg)
cv2.imwrite("messiLPF + ronaldoHPF.jpg", addedImages)
plotImg(addedImages)



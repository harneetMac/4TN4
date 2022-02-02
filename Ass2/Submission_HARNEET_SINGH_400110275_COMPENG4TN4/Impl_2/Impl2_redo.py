import cv2
import numpy as np
from matplotlib import pyplot as plt

#read images
messiImg = cv2.imread('messi.jpg', cv2.IMREAD_GRAYSCALE)
ronaldoImg = cv2.imread('ronaldo.jpg', cv2.IMREAD_GRAYSCALE)

#convert messi's image to frequency spectrum
dftMessi = cv2.dft(np.float32(messiImg), flags = cv2.DFT_COMPLEX_OUTPUT )
dft_shiftMessi = np.fft.fftshift(dftMessi)
magnitudephaseMessi = list(cv2.cartToPolar(dft_shiftMessi[:,:,0], dft_shiftMessi[:,:,1]))

#convert messi's image to frequency spectrum
dftRonaldo = cv2.dft(np.float32(ronaldoImg), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shiftRonaldo = np.fft.fftshift(dftRonaldo)
magnitudephaseRonaldo  = list(cv2.cartToPolar(dft_shiftRonaldo[:,:,0], dft_shiftRonaldo[:,:,1]))

#swap phases of both frequency spectrums
magnitudephaseMessi[1], magnitudephaseRonaldo[1] = magnitudephaseRonaldo[1], magnitudephaseMessi[1]

#convert back to cartesian domain and save the result
cartMessi = cv2.polarToCart(magnitudephaseMessi[0], magnitudephaseMessi[1])
f_ishiftMessi = np.fft.ifftshift(cv2.merge(cartMessi))
img_backMessi = cv2.idft(f_ishiftMessi, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
cv2.imwrite("Impl2MessiMag+RonaldoPhase.jpg", img_backMessi)

cartRonaldo = cv2.polarToCart(magnitudephaseRonaldo[0], magnitudephaseRonaldo[1])
f_ishiftRonaldo = np.fft.ifftshift(cv2.merge(cartRonaldo))
img_backRonaldo = cv2.idft(f_ishiftRonaldo, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
cv2.imwrite("Impl2RonaldoMag+MessiPhase.jpg", img_backRonaldo)


#notice plot shows slightly different result & if you were to do imshow(), you won't get anything from it (BE CAREFUL with imshow())
plt.subplot(111), plt.imshow(img_backMessi, cmap = 'gray')
plt.title(f'Impl2MessiMag+RonaldoPhase'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(111), plt.imshow(img_backRonaldo, cmap = 'gray')
plt.title(f'Impl2Ronaldo+MagMessiPhase'), plt.xticks([]), plt.yticks([])
plt.show()

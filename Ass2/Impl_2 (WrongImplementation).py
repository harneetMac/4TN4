import cv2
import numpy as np
from matplotlib import pyplot as plt

messiImg = cv2.imread('messi.jpg', cv2.IMREAD_GRAYSCALE)
ronaldoImg = cv2.imread('ronaldo.jpg', cv2.IMREAD_GRAYSCALE)


def fftImgShift(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitudeSpectrum = 20 * np.log( np.abs(fshift) )
    phaseSpectrum = np.angle(fshift)
    return magnitudeSpectrum, phaseSpectrum

def plotfft(img, fftImg, nameImg):
    plt.subplot(211), plt.imshow(img, cmap = 'gray')
    plt.title(f'{nameImg} Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(212),plt.imshow(fftImg, cmap = 'gray')
    plt.title(f'{nameImg} Frequency Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

def fftInversed(fftImg):
    imgIshift = np.fft.ifftshift(fftImg)
    imgInversed = np.fft.ifft2(imgIshift)
    imgMag = np.abs(imgInversed)
    return imgMag

messiMagSpectrum, messiPhaseSpectrum = fftImgShift(messiImg)
ronaldoMagSpectrum, ronaldoPhaseSpectrum = fftImgShift(ronaldoImg)
plotfft(messiImg, messiMagSpectrum, 'Messi')
plotfft(ronaldoImg, ronaldoMagSpectrum, 'Ronaldo')

mix1 = np.multiply(messiMagSpectrum, ronaldoPhaseSpectrum)
mix2 = np.multiply(ronaldoMagSpectrum, messiPhaseSpectrum)
plotfft(messiImg, mix1, 'MessiMag+RonaldoPhase')
plotfft(ronaldoImg, mix2, 'RonaldoMag+MessiPhase')

mix1Mag = fftInversed(mix1)
mix2Mag = fftInversed(mix2)
plotfft(mix1, mix1Mag, 'MessiMag+RonaldoPhase')
plotfft(mix2, mix2Mag, 'RonaldoMag+MessiPhase')


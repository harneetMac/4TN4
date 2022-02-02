import cv2
import numpy as np

# reading the image as is and storing it in a variable
img = cv2.imread('img1.png', cv2.IMREAD_COLOR)
# print(img.shape)

gamma = 0.60

#Applying the gamma correction formula using numpy array
#Scaling down to apply the formula and then, scaling up
#to range between 0 and 255 (integer)
gammaCorrectedArray = np.array( ((img/255)**gamma) * 255, dtype = 'uint8' )
# print(gammaCorrectedArray.shape)
# print(gammaCorrectedArray)

cv2.imshow("img2_gamma", gammaCorrectedArray)

#if image processed correctly, it will be saved by pressing 's',
#otherwise quit the window
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("img2_gamma.png", gammaCorrectedArray)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

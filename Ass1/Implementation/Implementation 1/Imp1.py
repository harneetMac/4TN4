import cv2

# reading the image as is and storing it in a variable
img = cv2.imread('img1.png', cv2.IMREAD_UNCHANGED)
# print(img)

#show image in a window named "img1"
cv2.imshow("img1", img)

#press 's' if you want to save image, else press any other key
#later, destroy all the windows to free up the resources
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("img1_0.png", img)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

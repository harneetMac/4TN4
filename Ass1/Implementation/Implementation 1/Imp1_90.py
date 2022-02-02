import cv2

# reading the image as is and storing it in a variable
img = cv2.imread('img1.png', cv2.IMREAD_UNCHANGED)
# print(img)

#rotating image counterclockwise by 90 degrees
img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("img1", img)

#if image processed correctly, it will be saved by pressing 's',
#otherwise quit the window and free up the resources
if cv2.waitKey(0) == ord('s'):
    cv2.imwrite("img1_90.png", img)
    cv2.destroyAllWindows()
else:
    cv2.destroyAllWindows()

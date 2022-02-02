import cv2
import numpy as np

frame_messi = cv2.imread('messi.jpg')
print(frame_messi.shape)
# print(frame_messi)

empty_frame = np.empty_like(frame_messi, dtype = 'uint8')
print(empty_frame.shape)
stacked_frames = np.vstack((frame_messi, empty_frame))
print(stacked_frames.shape)


# print(frame_messi[0][0][0])
# empty_frame[0][0][0] = 100
# print(empty_frame[0][0][0])
# print(frame_messi)

# frame_ronaldo = cv2.imread('ronaldo.jpg')
# print(frame_ronaldo)
# print(frame_ronaldo.shape)

# ravel_frame = np.ravel(frame_messi)
# # print(ravel_frame)
# print(ravel_frame.shape)
# unravel_frame = np.resize(ravel_frame, frame_messi.shape)
# print(unravel_frame.shape)
# print(unravel_frame)



# cv2.imshow('frame', frame)
# cv2.waitKey(0)

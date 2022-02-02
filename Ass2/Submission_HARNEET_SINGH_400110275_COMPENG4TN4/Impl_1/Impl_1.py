import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
backgroundImage = np.empty(shape = (int(cap.get(4)), int(cap.get(3)), 3))
all_frames = np.empty(shape = (14, int(cap.get(4)), int(cap.get(3)), 3), dtype = 'uint8')
# print(len(all_frames))

def loadFramesAsNP():
    index = 0
    i = 0

    while cap.isOpened():
        index += 1
        ret, frame = cap.read()
        # print(frame.shape)

        if not ret:
            print(f"Can't receive frame. Exiting after {index} frames ...")
            break

        if index % 10 == 0:
            all_frames[i] = frame
            # print(all_frames[i].shape)
            # cv2.imshow('frames', all_frames[i])
            # cv2.waitKey(0)

            i += 1

        # cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def findMedian():
    medianValuedImage = np.median(all_frames, axis=0).astype(dtype = 'uint8')
    cv2.imshow('background_image', medianValuedImage)

    if cv2.waitKey(0) == ord('s'):
        cv2.imwrite("impl1Background.jpg", medianValuedImage)
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()

    return medianValuedImage

def VideoWithoutBackground(backgroundImage):
    cap = cv2.VideoCapture('video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #important to get the framesize right, otherwise it won't work
    outVideo = cv2.VideoWriter('impl1Video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frameNoBackground = cap.read()

        if not ret:
            print(f"Can't receive frame. Exiting after playing frames ...")
            break

        frameNoBackground = frameNoBackground.astype(np.int16)
        withoutBackground = (np.abs(frameNoBackground - backgroundImage)).astype(np.uint8)
        # Next line makes it work without casting to int16 and back to int8
        # withoutBackground = cv2.absdiff(frame2, backgroundImage)

        outVideo.write(withoutBackground)
        cv2.imshow('frameNoBackground', withoutBackground)
        if cv2.waitKey(1) == ord('q'):
            break

    outVideo.release()
    cap.release()
    cv2.destroyAllWindows()


loadFramesAsNP()
backgroundImage = findMedian()
VideoWithoutBackground(backgroundImage)



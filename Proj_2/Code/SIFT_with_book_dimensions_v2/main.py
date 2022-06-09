import os
import cv2
import numpy as np
import support_functions

# First, we need to read in the train images from the 'Train_Images' folder
PATH = '../../Train_Images'
train_images_name_list = os.listdir(PATH)
# print(train_images_name_list)
print(f'Number of train images: {len(train_images_name_list)}')

# To store the images, we will create a list to store the name of the book
# and to store the image itself
train_images = []
book_names = []


def resize_image(img, scale_percent=60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized_img


# we will loop through the train_images_name_list to get the images and book names
for i, name in enumerate(train_images_name_list):
    image = cv2.imread(f'{PATH}/{name}')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(gray_image.shape)
    gray_image = resize_image(gray_image)
    print(f'Train Image {i + 1} size: {gray_image.shape}')

    train_images.append(gray_image)

    # print(name)
    # splitting to get rid of .jpg extension
    book_name = name.split(".")[0]
    # print(book_name)

    book_names.append(book_name)

# print(train_images)
print(book_names)

# now, we need to find the key-points and descriptors of the train images
# this needs to be computed only once and then, we can use these to classify images
# orb = cv2.ORB_create(nfeatures=1000)
sift = cv2.SIFT_create(nfeatures=1000)


def find_descriptors_keypoints(images):
    kps_list = []
    des_list = []

    for image in images:
        frameHeight, frameWidth = image.shape

        mask = np.zeros_like(image, dtype='uint8')
        cv2.rectangle(mask, (0, frameHeight // 4), (frameWidth, frameHeight), 255, -1)
        cv2.rectangle(mask, ((frameWidth // 2) - 60, frameHeight - 70), ((frameWidth // 2) + 60, frameHeight - 20), 0,
                      -1)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)

        keypoints, descriptors = sift.detectAndCompute(image, mask)

        kps_list.append(keypoints)
        des_list.append(descriptors)

    return kps_list, des_list


keypoints_list, descriptors_list = find_descriptors_keypoints(train_images)


# print(keypoints_list)
# print(len(descriptors_list))
# print(len(keypoints_list))


# we can use OpenCVs method to view the keypoints on the images
def draw_keypoints(images, kps):
    i = 0

    for image, kp in zip(images, kps):
        # print(image.shape)
        # print(kp)
        image_with_keypoints = cv2.drawKeypoints(image, kp, None)

        cv2.imshow(f'{book_names[i]}', image_with_keypoints)
        # cv2.imwrite(f'gray_{book_names[i]}.jpg', image_with_keypoints)
        cv2.waitKey(1000)
        i += 1


# draw_keypoints(train_images, keypoints_list)


# we need to function to compare the train images descriptor with the camera image descriptors
def find_camera_and_compare_descriptors(cam_img, train_imgs_des, threshold=100):
    cam_kps, cam_des = sift.detectAndCompute(cam_img, None)
    book_name_index = -1

    # using Brute Force matcher to match the descriptors between train and camera images
    bf = cv2.BFMatcher()

    match_list = []

    try:
        for train_des in train_imgs_des:
            matches = bf.knnMatch(train_des, cam_des, k=2)

            # Apply ratio test
            good_matches = []

            for m, n in matches:
                if m.distance < (0.75 * n.distance):
                    good_matches.append([m])

            match_list.append(len(good_matches))

        print(f'Matches between different train images is: {match_list}')

    except len(match_list) == 0:
        print("Matcher function did NOT find matches")

    # find the max value in the match_list to get the book's name
    if len(match_list) != 0:
        max_matched_value = max(match_list)

        if max_matched_value > threshold:
            book_name_index = match_list.index(max_matched_value)

    return book_name_index


# open up the webcam feed
cap = cv2.VideoCapture(0)
commence_book_thickness_process = False

# set camera resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# contrast_value = cap.get(cv2.CAP_PROP_CONTRAST)
# print(f'Cam Contrast Value: {contrast_value}')


# once we have the keypoints and descriptors, we can try to test it against live feed
# from the camera
def commence_video_processing():
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, cam_img = cap.read()
        if not ret:
            print("Could not read the image from the camera")
            break

        cam_img_color = cam_img.copy()
        cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)

        book_idx = find_camera_and_compare_descriptors(cam_img, descriptors_list)

        if book_idx != -1:
            cv2.putText(cam_img_color, book_names[book_idx], (25, 25), cv2.FONT_HERSHEY_TRIPLEX,
                        1, (0, 0, 255), thickness=1)

        cv2.imshow("SIFT Processing", cam_img_color)
        pressed_key = cv2.waitKey(1) & 0xFF

        # if the pressed_key is 'q' exit the program
        # else begin the book thickness measurement process
        if pressed_key == ord('q'):
            # When everything done, destroy the SIFT processing window
            cv2.destroyWindow("SIFT Processing")
            return None
        elif pressed_key == ord('p'):
            global commence_book_thickness_process
            commence_book_thickness_process = True
            # When everything done, destroy the SIFT processing window
            cv2.destroyWindow("SIFT Processing")
            return cam_img_color


# start the video processing task to detect book using SIFT
# measure_thickness_image = commence_video_processing()

# TODO delete the following block later
_ = commence_video_processing()
# measure_thickness_image = cv2.imread('screenshot_a4_book.jpg')
# cv2.imwrite("screenshot_a4_book.jpg", measure_thickness_image)


#############################################################################
#############################################################################

# This is where task 2 begins that calculates the book measurements

# A4 size paper is used to determine the physical contour dimensions
scale_factor = 3

width_A4_paper = 210 * scale_factor
height_A4_paper = 297 * scale_factor
elevation_bias = 0.9                  # because of book thickness
page_scale = 160


def book_measurements():
    # open up the webcam feed
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, measure_thickness_image = cap.read()
        if not ret:
            print("Could not read the image from the camera")
            break

        img_contours, contour_info = support_functions.get_contours(measure_thickness_image,
                                                                    min_area=50_000,
                                                                    filtered_obj=4,
                                                                    display_canny=False,
                                                                    display_contours=False)

        # if contour list is not empty then we need to find the contour of A4 paper and the book
        if len(contour_info) != 0:
            largest_contour_pts = contour_info[0][2]
            # print(f'Largest Contour Area Points of shape {largest_contour_pts.shape}: \n{largest_contour_pts}')

            # now, we can warp the image
            img_warped = support_functions.get_warped_img(img_contours,
                                                          largest_contour_pts,
                                                          width_A4_paper,
                                                          height_A4_paper)
            x_A4, y_A4, w_A4, h_A4 = contour_info[0][3]
            # print(f'{x_A4=}, {y_A4=}, {w_A4=}, {h_A4=}, area_A4 = {w_A4 * h_A4}')
            # cv2.imshow("Warped Image", img_warped)
            # cv2.imwrite("warped_image.jpg", img_warped)
            # cv2.waitKey(0)

            # from hereon, we can find the internal contours which is the book
            # print("Internal Contour Process Begins")
            img_internal_cnt, books_contour_pts = support_functions.get_contours(img_warped,
                                                                                 min_area=25_000,
                                                                                 filtered_obj=4,
                                                                                 display_canny=False,
                                                                                 display_contours=False)

            if len(books_contour_pts) != 0:
                for cnt in books_contour_pts:
                    cv2.polylines(img_internal_cnt, [cnt[2]], True, (0, 255, 0), thickness=2)

                    # # Method 1: get rectangle around the contour
                    # book_rect = cv2.minAreaRect(cnt[4])
                    # (x, y), (w, h), angle = book_rect
                    # book_width_v1 = round((w//scale_factor) / 10, 1)
                    # book_height_v1 = round((h//scale_factor) / 10, 1)
                    # print(f'Method 1: Book Width: {book_width_v1} cm, height: {book_height_v1} cm')

                    # Method 2:
                    new_points = support_functions.rearrange_contour_points(cnt[2])
                    book_width = support_functions.find_distance_corner_pts(new_points[0][0] // scale_factor,
                                                                            new_points[1][0] // scale_factor)
                    book_height = support_functions.find_distance_corner_pts(new_points[0][0] // scale_factor,
                                                                             new_points[2][0] // scale_factor)
                    book_width = round((book_width / 10) * elevation_bias, 1)
                    book_height = round((book_height / 10) * elevation_bias, 1)

                    # print(f'Method 2: Book Width: {book_width} cm, height: {book_height} cm')

                    # put text on the image
                    cv2.arrowedLine(img_internal_cnt, (new_points[0][0][0], new_points[0][0][1]),
                                    (new_points[1][0][0], new_points[1][0][1]),
                                    (0, 0, 0), 3, 8, 0, 0.05)

                    cv2.arrowedLine(img_internal_cnt, (new_points[0][0][0], new_points[0][0][1]),
                                    (new_points[2][0][0], new_points[2][0][1]),
                                    (0, 0, 0), 3, 8, 0, 0.05)

                    x, y, w, h = cnt[3]
                    # print(f'{x=}, {y=}, {w}, {h=}, area = {w * h}')
                    cv2.putText(img_internal_cnt, f'{book_width} cm', (x + w // 2, y - 10),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)
                    cv2.putText(img_internal_cnt, f'{book_height} cm', (x - 75, (y + h) // 2),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)

                    if book_width < 5:
                        thickness = int(book_width * 160)
                        cv2.putText(img_internal_cnt, f'{thickness} pages', (x, y + h // 4),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 0), 2)


                # book_contour_pts = books_contour_pts[0][2]
                # print(f'Book Contour Area Points of shape {book_contour_pts.shape}: \n{book_contour_pts}')

            cv2.imshow("Books Contour on the warped image", img_internal_cnt)

        pressed_key = cv2.waitKey(1)
        # pressed_key = cv2.waitKey(1)
        #
        # if the pressed_key is 'q' exit the program
        if pressed_key == ord('q'):
            break


if commence_book_thickness_process:
    print(f'\nMeasure of book thickness process begins')

    # cv2.imshow("Book Measurement Frame", measure_thickness_image)
    # cv2.imwrite("Web Cam Frame.jpg", measure_thickness_image)
    # cv2.waitKey(0)

    book_measurements()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



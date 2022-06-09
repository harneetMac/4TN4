import cv2
import numpy as np


# get canny edge detectors optimal threshold
def get_canny_thresholds(img):
    # based on sigma value and median of the image, low and high threshold will be
    # calculated. The idea is to pick values that are close to the median.
    sigma = 2
    median = np.median(img)
    lowThreshold = int(max(0, (1.0 - sigma) * median))
    highThreshold = int(min(255, (1.0 + sigma) * median))

    return lowThreshold, highThreshold


def get_contours(img, min_area=20000, filtered_obj=0, display_canny=False, display_contours=False, canny_threshold=None):
    # change image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur operation - preprocessing image using gaussian filter
    kernelSizeGaussian = 5
    gaussianFiltered_img = cv2.GaussianBlur(img_gray, (kernelSizeGaussian, kernelSizeGaussian),
                                            0, borderType=cv2.BORDER_CONSTANT)

    # cv2.imwrite("gaussian_filtered_frame.jpg", gaussianFiltered_img)


    # determine canny threshold values
    if canny_threshold is None:
        lowThreshold, highThreshold = get_canny_thresholds(gaussianFiltered_img)
        # print(f'Canny low threshold = {lowThreshold}, high threshold = {highThreshold}')
    else:
        # receives a list of low and high threshold i.e. [low, high]
        lowThreshold, highThreshold = canny_threshold

    # calling Canny() function to get the license plate only
    padding = 5
    cannyOperated_img = cv2.Canny(gaussianFiltered_img, lowThreshold, highThreshold,
                                  None, L2gradient=True)
    cannyOperated_img = cannyOperated_img[padding:cannyOperated_img.shape[0] - padding,
                        padding:cannyOperated_img.shape[1] - padding]

    # cv2.imwrite("canny_edge_detection.jpg", cannyOperated_img)

    # dilate and erode canny image
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSizeGaussian, kernelSizeGaussian))
    img_dilated = cv2.dilate(cannyOperated_img, morph_kernel, iterations=3)
    img_eroded = cv2.erode(img_dilated, morph_kernel, iterations=2)

    if display_canny:
        cv2.imshow("Canny Image", img_eroded)
        cv2.imwrite("canny+morph_dilate_erode.jpg", img_eroded)
        cv2.waitKey(0)

    # now find contours from binarized image
    contours, _ = cv2.findContours(img_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_info = []

    for cnt in contours:
        area = cv2.cv2.contourArea(cnt)
        # print(f'contour area is: {area}')

        # if area is above a certain number (empirically found)
        if area > min_area:
            # print(f'contour area is: {area}')

            # find the perimeter of the contour
            perimeter = cv2.arcLength(cnt, True)

            # find approximate shape of the contour - we're interested in rectangle i.e. 4 points
            epsilon = 0.01 * perimeter
            approx_shape = cv2.approxPolyDP(cnt, epsilon, True)
            bbox = cv2.boundingRect(approx_shape)

            # we require 4 points to find rectangle in the image
            if filtered_obj > 0:
                number_points = len(approx_shape)
                # only proceed if a shape is found in the image i.e. 4 corner points of rectangle
                if number_points == filtered_obj:
                    contour_info.append([len(approx_shape), area, approx_shape, bbox, cnt])

            else:
                contour_info.append([len(approx_shape), area, approx_shape, bbox, cnt])

    # sort the contour_info list based on the area
    contour_info = sorted(contour_info, key=lambda i: i[1], reverse=True)
    # print(f'contour info: \n{contour_info}')
    # return None

    if display_contours:
        for cnt in contour_info:
            cv2.drawContours(img, cnt[4], -1, (0, 0, 255), thickness=3)
            # cv2.polylines(img, cnt[4], True, (0, 255, 0), thickness=3)

            # # get rectangle around the contour
            # rect = cv2.minAreaRect(cnt[4])
            # (x, y), (w, h), angle = rect

            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # #
            # cv2.polylines(img, [box], True, (0, 255, 0), thickness=2)

            cv2.imshow("Contours on image", img)
            cv2.imwrite("contours.jpg", img)
            cv2.waitKey(0)

    return img, contour_info


# note that the contour points are not always in the same order i.e. top-left, top-right,
# bottom-left and bottom left are flipped for different contours
# so we need to fix the pattern to always warp in the correct order
def rearrange_contour_points(cnt_pts):
    new_pts = np.zeros_like(cnt_pts)

    # we need to reshape the contour points from (4,1,2) to (4,2)
    cnt_pts = cnt_pts.reshape((4, 2))

    # image contour rectangle of corner points: (1,1), (1,5), (5,1), (5,5)
    # largest sum of (x,y) will get us the bottom-left corner e.g. (5+5) = 10
    # smallest sum of (x,y) will get us the top-left corner e.g. (1+1) = 2
    # largest difference (x,y) will get us the top-right corner e.g. (5-1) = 4
    # smallest difference (x,y) will get us the bottom-left corner e.g. (1-5) = -4

    add_x_y = cnt_pts.sum(axis=1)
    # top-left point
    new_pts[0] = cnt_pts[np.argmin(add_x_y)]
    # bottom-right point
    new_pts[3] = cnt_pts[np.argmax(add_x_y)]

    diff_x_y = np.diff(cnt_pts, axis=1)
    # top-right point
    new_pts[1] = cnt_pts[np.argmin(diff_x_y)]
    # bottom-left point
    new_pts[2] = cnt_pts[np.argmax(diff_x_y)]

    return new_pts


def get_warped_img(img, cnt_pts, width, height, padding=15):
    # print(f'Contour Points are: \n{cnt_pts}')

    cnt_pts = rearrange_contour_points(cnt_pts)
    # print(f'Rearranged Contour Points are: \n{cnt_pts}')

    input_pts = np.float32(cnt_pts)

    # creating output points to warp image
    output_pts = np.float32([[0, 0],
                             [width, 0],
                             [0, height],
                             [width, height]])

    # creating transformation matrix to warp the perspective
    transformation_matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
    img_warped = cv2.warpPerspective(img, transformation_matrix, (width, height))
    img_warped = img_warped[padding:img_warped.shape[0] - padding, padding:img_warped.shape[1] - padding]

    return img_warped


# find distance between the corner points
def find_distance_corner_pts(pt_1, pt_2):
    return ((pt_2[0]-pt_1[0])**2 + (pt_2[1]-pt_1[1])**2)**0.5

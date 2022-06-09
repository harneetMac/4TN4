import cv2
import numpy as np
import matplotlib.pyplot as plt

# read given images
img21Color = cv2.imread("image2_1.jpg")
img22Color = cv2.imread("image2_2.jpg")
img23Color = cv2.imread("image2_3.jpg")
img24Color = cv2.imread("image2_4.jpg")
img1 = cv2.cvtColor(img21Color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img22Color, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img23Color, cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img24Color, cv2.COLOR_BGR2GRAY)


# this function will use SIFT to find keypoints and descriptors
def detectAndDescribe(img):
    descriptor = cv2.SIFT_create()

    # get keypoints and descriptors
    kps, features = descriptor.detectAndCompute(img, None)
    return kps, features


# getting the keypoints and features of all the images
kps1, features1 = detectAndDescribe(img1)
kps2, features2 = detectAndDescribe(img2)
kps3, features3 = detectAndDescribe(img3)
kps4, features4 = detectAndDescribe(img4)
# print(f'feature shape of feature 1 is {features1.shape}')
# print(f'{features1=}')


# function is used to display keypoints on the image (shows gradient: strength and orientation)
def drawKeyPoints(img, kps):
    img = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# calling functions to draw gradients for all images
img1_kps = drawKeyPoints(img1, kps1)
img2_kps = drawKeyPoints(img2, kps2)
img3_kps = drawKeyPoints(img3, kps3)
img4_kps = drawKeyPoints(img4, kps4)

# plotting keypoints on the images
# plt.subplot(221), plt.imshow(img1_kps, cmap='gray', vmin=0, vmax=255)
# plt.title('img1_kps'), plt.xticks([]), plt.yticks([])
# plt.subplot(222), plt.imshow(img2_kps, cmap='gray', vmin=0, vmax=255)
# plt.title('img2_kps'), plt.xticks([]), plt.yticks([])
# plt.subplot(223), plt.imshow(img3_kps, cmap='gray', vmin=0, vmax=255)
# plt.title('img3_kps'), plt.xticks([]), plt.yticks([])
# plt.subplot(224), plt.imshow(img4_kps, cmap='gray', vmin=0, vmax=255)
# plt.title('img4_kps'), plt.xticks([]), plt.yticks([])
# plt.show()

# creating matcher object to be used in matching features of two images
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


# function matches the features between two images and
# computes distance between the features
def bruteForceMatcher(feat1, feat2):
    # Match descriptors.
    matches = bf.match(feat1, feat2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

# calling function to match features between all pairs of images
matchesImg12 = bruteForceMatcher(features1, features2)
matchesImg13 = bruteForceMatcher(features1, features3)
matchesImg14 = bruteForceMatcher(features1, features4)
matchesImg23 = bruteForceMatcher(features2, features3)
matchesImg24 = bruteForceMatcher(features2, features4)
matchesImg34 = bruteForceMatcher(features3, features4)


# function is used to plot the matched points on two images
def drawMatchedImages(matches, image1, kp1, image2, kp2):
    # Draw first 100 matches.
    matchedImg = cv2.drawMatches(image1, kp1, image2, kp2, matches[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(20, 10))
    plt.imshow(matchedImg)
    plt.show()


# calling function to visualize matching points between all images in pairs
# drawMatchedImages(matchesImg12, img1, kps1, img2, kps2)
# drawMatchedImages(matchesImg13, img1, kps1, img3, kps3)
# drawMatchedImages(matchesImg14, img1, kps1, img4, kps4)
# drawMatchedImages(matchesImg23, img2, kps2, img3, kps3)
# drawMatchedImages(matchesImg24, img2, kps2, img4, kps4)
# drawMatchedImages(matchesImg34, img3, kps3, img4, kps4)


# function gets the homography matrix to be able to transform images over each other
def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        H, stat = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        return matches, H, stat
    else:
        return None

# getting homography matrix for each pair of images
# ideally, to reduce computation, we would only need to compute this step 3 times but
# to see the matches between all the images, I am getting H matrix for all the pairs
matches12, HomographyMat12, status12 = getHomography(kps1, kps2, features1, features2, matchesImg12, 4)
matches13, HomographyMat13, status13 = getHomography(kps1, kps3, features1, features3, matchesImg13, 4)
matches14, HomographyMat14, status14 = getHomography(kps1, kps4, features1, features4, matchesImg14, 4)
matches23, HomographyMat23, status23 = getHomography(kps2, kps3, features2, features3, matchesImg23, 4)
matches24, HomographyMat24, status24 = getHomography(kps2, kps4, features2, features4, matchesImg24, 4)
matches34, HomographyMat34, status34 = getHomography(kps3, kps4, features3, features4, matchesImg34, 4)
# print(f'Homography Matrix_12 is: \n{HomographyMat12}')
# print(f'Homography Matrix_13 is: \n{HomographyMat13}')
# print(f'Homography Matrix_14 is: \n{HomographyMat14}')
# print(f'Homography Matrix_23 is: \n{HomographyMat23}')
# print(f'Homography Matrix_24 is: \n{HomographyMat24}')
# print(f'Homography Matrix_34 is: \n{HomographyMat34}')


# using H matrix, this function is warping two images together to create panaroma
def warpImages(image1, image2, H_Mat):
    rows1, cols1 = image1.shape[:2]
    rows2, cols2 = image2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H_Mat)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    # print(H_translation)

    outputImg = cv2.warpPerspective(image1, H_translation.dot(H_Mat), (x_max - x_min, y_max - y_min))
    outputImg[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = image2

    return outputImg

# calling warpImages to warp each pair of image (ideally, we would only need 2 but I wanted to see the results)
warpImages12 = warpImages(img21Color, img22Color, HomographyMat12)
warpImages13 = warpImages(img21Color, img23Color, HomographyMat13)
warpImages14 = warpImages(img21Color, img24Color, HomographyMat14)
warpImages23 = warpImages(img22Color, img23Color, HomographyMat23)
warpImages24 = warpImages(img22Color, img24Color, HomographyMat24)
warpImages34 = warpImages(img23Color, img24Color, HomographyMat34)

# plt.subplot(231), plt.imshow(warpImages12, cmap='gray', vmin=0, vmax=255)
# plt.subplot(232), plt.imshow(warpImages13, cmap='gray', vmin=0, vmax=255)
# plt.subplot(233), plt.imshow(warpImages14, cmap='gray', vmin=0, vmax=255)
# plt.subplot(234), plt.imshow(warpImages23, cmap='gray', vmin=0, vmax=255)
# plt.subplot(235), plt.imshow(warpImages24, cmap='gray', vmin=0, vmax=255)
# plt.subplot(236), plt.imshow(warpImages34, cmap='gray', vmin=0, vmax=255)
# plt.show()

# finally, creating full image using top 2 warped images and bottom 2 warped images
kps12, features12 = detectAndDescribe(warpImages12)
kps34, features34 = detectAndDescribe(warpImages34)
matchesImg1234 = bruteForceMatcher(features12, features34)
matches1234, HomographyMat1234, status1234 = getHomography(kps12, kps34, features12, features34, matchesImg1234, 4)
warpImages1234 = warpImages(warpImages12, warpImages34, HomographyMat1234)
warpImages1234 = cv2.cvtColor(warpImages1234, cv2.COLOR_BGR2RGB)
# plt.imshow(warpImages1234, cmap='gray', vmin=0, vmax=255)
plt.imshow(warpImages1234)
plt.show()

# cv2.imshow("image 1", img1)
# cv2.imshow("image 2", img2)
# cv2.imshow("image 3", img3)
# cv2.imshow("image 4", img4)
# cv2.waitKey(0)

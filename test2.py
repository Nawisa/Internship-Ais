# import the necessary packages
from turtle import left
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from skimage.metrics import structural_similarity

############################# Method to find the mid point######################################


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("nw2.jpg")
image = cv2.resize(image, (600, 350))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)

# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 0, 250)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv2() else cnts[0]

# loop over the contours individually
for c in cnts:
    # This is to ignore that small hair countour which is not big enough
    if cv2.contourArea(c) < 500:
        continue

    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    # draw the contours on the image
    orig = image.copy()
    # cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 3)

    # # unpack the ordered bounding box, then compute the midpoint
    # # between the top-left and top-right coordinates, followed by
    # # the midpoint between bottom-left and bottom-right coordinates
    # (tl, tr, br, bl) = box
    # (tltrX, tltrY) = midpoint(tl, tr)
    # (blbrX, blbrY) = midpoint(bl, br)

    # # compute the midpoint between the top-left and top-right points,
    # # followed by the midpoint between the top-righ and bottom-right
    # (tlblX, tlblY) = midpoint(tl, bl)
    # (trbrX, trbrY) = midpoint(tr, br)

    # # draw and write the midpoints on the image
    # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    # cv2.putText(orig, "({},{})".format(tltrX, tltrY), (int(tltrX - 50), int(tltrY - 10) - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    # cv2.putText(orig, "({},{})".format(blbrX, blbrY), (int(blbrX - 50), int(blbrY - 10) - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    # cv2.putText(orig, "({},{})".format(tlblX, tlblY), (int(tlblX - 50), int(tlblY - 10) - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # cv2.putText(orig, "({},{})".format(trbrX, trbrY), (int(trbrX - 50), int(trbrY - 10) - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # draw lines between the midpoints
    # image = cv2.line(Img, start_point, end_point, color, thickness)
    # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
    # (255, 0, 255), 2)

################# separate left and right sides ################################
h, w, channels = orig.shape

half = w//2

# # this will be the first column
left_part = orig[:, :half]

# # [:,:half] means all the rows and
# # all the columns upto index half

# # this will be the second column
right_part = orig[:, half:]

# # Grayscale
gray1 = cv2.cvtColor(left_part, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(right_part, cv2.COLOR_BGR2GRAY)

# # Find the difference between the two images
# # Calculate absolute difference between two arrays
(similar, diff) = structural_similarity(gray1, gray2, full=True)
diff = (diff*255).astype("uint8")
# diff = cv2.absdiff(gray1, gray2)
cv2.imshow("diff(img1, img2)", diff)


################## calculate contour area #############################

# # Apply threshold. Apply both THRESH_BINARY and THRESH_OTSU
thresh = cv2.threshold(
    diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow("Threshold", thresh)

# # # Dilation
# kernel = np.ones((5, 5), np.uint8)
# dilate = cv2.dilate(thresh, kernel, iterations=2)
# # cv2.imshow("Dilate", dilate)

# # Calculate contours
contours = cv2.findContours(
    thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# # determine area
for contour in contours:
    # if cv2.contourArea(box) > 10000:
    if cv2.contourArea(contour) > 400 and cv2.contourArea(contour) < 500:
        # img_height = orig.shape[0]
        # Calculate bounding box around contour
        x, y, w, h = cv2.boundingRect(contour)
        # frameArea = (w*h)
        # if cv2.boundingRect(frameArea) < 500:
        # areaTH = frameArea
        # Draw rectangle - bounding box on both images
        cv2.rectangle(left_part, (x+w, y+h), (w+y, h), (0, 0, 255), 2)
        # cv2.rectangle(right_part, (x, y+w), (w+x, h+w), (0, 0, 255), 2)

    # # Show images with rectangles on differences
    # x = np.zeros((channels, 10, 3), np.uint8)
result = np.hstack((left_part, right_part))
cv2.imshow("Differences", result)

# # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
# #          (255, 0, 255), 2)

# # compute the Euclidean distance between the midpoints
# # dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
# # dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

# # loop over the original points
# for (xA, yA) in list(box):
#     # draw circles corresponding to the current points and
#     cv2.circle(orig, (int(xA), int(yA)), 5, (0, 0, 255), -1)
#     cv2.putText(orig, "({},{})".format(xA, yA), (int(xA - 50), int(yA - 10) - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#     # show the output image, resize it as per your requirements
cv2.imshow("Image", orig)
cv2.waitKey(0)

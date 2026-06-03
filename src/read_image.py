import cv2 as cv
import sys

img = cv.imread("src//image1.jpg",cv.IMREAD_COLOR_BGR) #IMREAD_COLOR, IMREAD_UNCHANGED , IMREAD_GRAYSCALE

if img is None:
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
cv.resizeWindow("Display window", (img.shape[1],img.shape[0]))
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("starry_night.png", img)
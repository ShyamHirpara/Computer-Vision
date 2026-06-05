import cv2 as cv
import numpy as np 
import os

cap = cv.VideoCapture(0)
dir_path = "src\\captures\\"

fourcc = cv.VideoWriter_fourcc(*'XVID') 
fps = 20.0
frame_size = (640, 480)
out_bgr = cv.VideoWriter(os.path.join(dir_path,'BGR_video.avi'), fourcc, fps, frame_size)
out_gray = cv.VideoWriter(os.path.join(dir_path,'gray_video.avi'), fourcc, fps, frame_size, isColor = False)
out_hsv = cv.VideoWriter(os.path.join(dir_path,'HSV_video.avi'), fourcc, fps, frame_size)


while True: 
    ret, frame = cap.read()

    if not ret :
        print("frame can't be received")
        break

    frame = cv.flip(frame,1)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)


    cv.imshow('color frame',frame)
    cv.imshow("gray frame",gray)
    cv.imshow("hsv frame",hsv)

    out_bgr.write(frame)
    out_gray.write(gray)
    out_hsv.write(hsv)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
out_bgr.release()
out_gray.release()
out_hsv.release()
cv.destroyAllWindows()


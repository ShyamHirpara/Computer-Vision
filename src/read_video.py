import cv2 as cv 
import numpy as np  
import os

path = 'src//captures//'
cap = cv.VideoCapture(path+'output.avi')

if not cap.isOpened():
    print("can't open video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret :
        print("frame can't be received")
        break
    
    cv.imshow('video',frame)

    key = cv.waitKey(int(1000/cap.get(cv.CAP_PROP_FPS)))
    
    if key == ord('q'):
        break
    

cap.release()
cv.destroyAllWindows()
    
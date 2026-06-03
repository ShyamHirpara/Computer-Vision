import cv2 as cv 
import numpy as np  
import os

cap = cv.VideoCapture(0)
counter = 0
path = 'src//captures//'
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter(path+'output.avi', fourcc, 20.0, (640, 480), isColor=True)
out_gray = cv.VideoWriter(path+'output_gray.avi', fourcc, 20.0, (640, 480), isColor=False)

if not cap.isOpened():
    print('cannot open camera')
    exit()

while True:
    ret, frame = cap.read()

    if not ret :
        print("frame can't be received")
        break
    frame =cv.flip(frame,1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


    cv.imshow('frame',gray)
    cv.imshow("color frame" ,frame)
    out.write(frame)
    out_gray.write(gray)

    key = cv.waitKey(1)
    
    if key == ord('s'):
        cv.imwrite(path+'image'+'_gray'+str(counter) + '.jpg',gray)
        cv.imwrite(path+'image'+str(counter) + '.jpg',frame)
        counter += 1
    elif key == ord('q'):
        break
    

cap.release()
out.release()
out_gray.release()
cv.destroyAllWindows()
# delete all images from image0.jpg to image<counter>.jpg and image0_gray.jpg to image<counter>_gray.jpg
for i in range(counter):
    os.remove(path+'image' + str(i) + '.jpg')
    os.remove(path+'image' + '_gray' + str(i) + '.jpg')
    
import cv2 as cv
import numpy as np


# events = [i for i in dir(cv) if 'EVENT' in i] # uncomment here for listing all mouse events 
# print( events )
""" ['EVENT_FLAG_ALTKEY', 
     'EVENT_FLAG_CTRLKEY', 
     'EVENT_FLAG_LBUTTON', 
     'EVENT_FLAG_MBUTTON', 
     'EVENT_FLAG_RBUTTON', 
     'EVENT_FLAG_SHIFTKEY', 
     'EVENT_LBUTTONDBLCLK', 
     'EVENT_LBUTTONDOWN', 
     'EVENT_LBUTTONUP', 
     'EVENT_MBUTTONDBLCLK', 
     'EVENT_MBUTTONDOWN', 
     'EVENT_MBUTTONUP', 
     'EVENT_MOUSEHWHEEL', 
     'EVENT_MOUSEMOVE', 
     'EVENT_MOUSEWHEEL', 
     'EVENT_RBUTTONDBLCLK', 
     'EVENT_RBUTTONDOWN', 
     'EVENT_RBUTTONUP'] """


# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.rectangle(img, (0,0),(511,511), (0,0,0),-1)
        cv.circle(img,(x,y),50,(0,0,255),2)
    if event == cv.EVENT_LBUTTONUP:
        cv.circle(img,(x,y),50,(255,0,0),2)

# function for drawing a line
def draw_line(event,x,y, flags, param):
    global a,b,c, color, thickness, drawing, font,size 
    color=(0,0,255)
    thickness = 2
    size = 0.5
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        cv.rectangle(img,(0,0),(512,512),(0,0,0),-1)
        a = (x,y)
        cv.putText(img, f"({a[0]},{a[1]})", a, font, size, (255, 255, 255), 1, cv.LINE_AA)  
    
    if event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            b = (x,y)
            cv.rectangle(img,(0,0),(512,512),(0,0,0),-1)
            cv.line(img,a,b,color,thickness)
            cv.putText(img, f"({a[0]},{a[1]})", a, font, size, (255, 255, 255), 1, cv.LINE_AA)  
            cv.putText(img, f"({b[0]},{b[1]})", b, font, size, (255, 255, 255), 1, cv.LINE_AA)  

    if event == cv.EVENT_LBUTTONUP:
        drawing = False
        c = (x,y)
        cv.line(img,a,c,color,thickness) 
        cv.putText(img, f"({a[0]},{a[1]})", a, font, size, (255, 255, 255), 1, cv.LINE_AA)  
        cv.putText(img, f"({c[0]},{c[1]})", c, font, size, (255, 255, 255), 1, cv.LINE_AA)     
    

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_line)
drawing = False
font = cv.FONT_HERSHEY_SIMPLEX
 
while(1):
    cv.imshow('image',img)
    if cv.waitKey(20) == ord('x'):
        break
cv.destroyAllWindows()
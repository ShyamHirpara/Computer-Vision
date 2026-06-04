import cv2 as cv 
import numpy as np
# Create a blank black canvas: 512x512 pixels, 3 color channels (BGR), 8-bit unsigned int
img = np.zeros((512, 512, 3), np.uint8)

# ========================== Section: Drawing a Line ==========================
# cv.line(image, start_point, end_point, color, thickness)
#   image     : the canvas to draw on
#   start_point / end_point : (x, y) pixel coordinates of the line endpoints
#   color     : BGR tuple — (Blue, Green, Red), values 0-255
#   thickness : line width in pixels

# Draw a blue diagonal line from top-left area to bottom-right area
a, b, color, thickness = (128, 128), (384, 384), (255, 0, 0), 2  # BGR (255,0,0) = Blue
cv.line(img, a, b, color, thickness)

# Draw a red diagonal line from bottom-left area to top-right area (forms an X with above)
a, b, color, thickness = (128, 384), (384, 128), (0, 0, 255), 2  # BGR (0,0,255) = Red
cv.line(img, a, b, color, thickness)

cv.imshow("Image", img)  # display the image in a window
cv.waitKey(0)            # wait indefinitely until any key is pressed

# ========================== Section: Drawing a Rectangle ==========================
# cv.rectangle(image, start_point, end_point, color, thickness)
#   image     : the canvas to draw on 
#   start_point / end_point : (x, y) pixel coordinates of the rectangle's top-left and bottom-right corners
#   color     : BGR tuple — (Blue, Green, Red), values 0-255
#   thickness : line width in pixels

cv.rectangle(img, a,b, color, thickness)

cv.imshow("Image", img) 
cv.waitKey(0)          

# ========================== Section: Drawing a Circle ==========================
# cv.circle(image, center_point, radius, color, thickness)
#   image     : the canvas to draw on
#   center_point : (x, y) pixel coordinates of the circle's center
#   radius    : radius of the circle in pixels
#   color     : BGR tuple — (Blue, Green, Red), values 0-255
#   thickness : line width in pixels

a,color, radius, thickness = (256,256),(0,255,0), 128,2
cv.circle(img, a, radius, color, thickness)

cv.imshow("Image", img) 
cv.waitKey(0)         

# ========================== Section: Drawing a Ellipse ==========================
# cv.ellipse(image, center, (major_axis, minor_axis), rotation_angle, start_angle, end_angle, color, thickness)
#   image     : the canvas to draw on
#   center : (x, y) pixel coordinates of the ellipse's center
#   major_axis : radius along the major axis in pixels
#   minor_axis : radius along the minor axis in pixels
#   rotation_angle : rotation of the ellipse in degrees
#   start_angle : starting angle of the ellipse in degrees
#   end_angle : ending angle of the ellipse in degrees
#   color : BGR tuple — (Blue, Green, Red), values 0-255
#   thickness : line width in pixels
center , major_axis, minor_axis, rotation_angle, start_angle, end_angle, color, thickness = (256,256),128,64,180,0,360,(255,0,0),2
cv.ellipse(img, center, (major_axis, minor_axis), rotation_angle, start_angle, end_angle, color, thickness)
cv.imshow("Image", img) 
cv.waitKey(0)         

# ========================== Section: Drawing a Polygon ==========================
# cv.polylines(image, points, is_closed, color, thickness)
#   image     : the canvas to draw on
#   points    : array of (x, y) pixel coordinates of the polygon's vertices
#   is_closed : whether the polygon is closed (True = connect last vertex to first)
#   color     : BGR tuple — (Blue, Green, Red), values 0-255
#   thickness : line width in pixels

points = np.array([(128, 128), (256, 128), (384, 256), (256, 384), (128, 256)], np.int32)
is_closed = True
color = (128, 0, 128)
thickness = 2
cv.polylines(img, [points], is_closed, color, thickness)
cv.imshow("Image", img)
cv.waitKey(0)

# ========================== Section: Adding Text to Images ==========================
# font = cv.FONT_HERSHEY_SIMPLEX
# cv.putText(img, text, (x, y), font, font_scale, color, thickness)
#   img       : the canvas to draw on
#   text      : the text to draw
#   (x, y)    : the coordinates where the text should start
#   font      : the font type
#   font_scale: the size of the font
#   color     : BGR tuple — (Blue, Green, Red), values 0-255
#   thickness : the thickness of the text

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, "Shyam Hirpara", (128, 128), font, 1, (255, 255, 255), 1, cv.LINE_AA)
cv.imshow("Image", img)
cv.waitKey(0)   

cv.destroyAllWindows()  
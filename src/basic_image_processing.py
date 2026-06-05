"""
basic_image_processing.py
--------------------------
Demonstrates fundamental image operations using OpenCV:
  - Reading image properties
  - Region of Interest (ROI) extraction and manipulation
  - Splitting color channels
  - Adding borders with different border types

Reference: https://docs.opencv.org/4.x/d3/df2/tutorial_py_basic_ops.html
"""

import cv2 as cv
import numpy as np

# ─────────────────────────────────────────────
# 1. Load Image & Print Basic Properties
# ─────────────────────────────────────────────

# Directory path where images are stored
dir_path = "src\\captures\\"

# Read the image from disk into a NumPy array (BGR format by default)
img = cv.imread(dir_path + "messi5.jpg")

# Abort if the image could not be loaded (e.g., wrong path or missing file)
assert img is not None, "file could not be read, check with os.path.exists()"

# Print image dimensions: (height, width, channels)
print(img.shape)

# Print total number of pixels × channels (height × width × channels)
print(f"size : {img.size}")

# Print the data type of each pixel value (e.g., uint8 for 0–255 range)
print(f"type : {img.dtype}")

# ─────────────────────────────────────────────
# 2. Region of Interest (ROI) — Extraction
# ─────────────────────────────────────────────

# Crop a 200×200 pixel patch from the image (rows 100–300, cols 100–300)
# This could represent a face or any object of interest
face = img[0:100,0:100]

# Confirm the dimensions of the extracted patch
print(f"size of face : {face.shape}")

# ─────────────────────────────────────────────
# 3. ROI Manipulation — Copy Patch to New Location
# ─────────────────────────────────────────────

# Paste the cropped face patch into a different region of the image
# Places it at rows 400–600, cols 100–300 (lower-left area)
img[100:200,100:200] = face

# Display the modified image with the pasted ROI
cv.imshow("Image", img)

# Wait for a key press before continuing
cv.waitKey(0)

# ─────────────────────────────────────────────
# 4. Split Color Channels (BGR → B, G, R)
# ─────────────────────────────────────────────

# OpenCV stores images in BGR order (not RGB)
# Index 0 = Blue, 1 = Green, 2 = Red channel
b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

# Display each channel as a grayscale image (higher = more of that color)
cv.imshow("Blue", b)
cv.imshow("Green", g)
cv.imshow("Red", r)

# Wait for a key press before moving to border examples
cv.waitKey(0)

# ─────────────────────────────────────────────
# 5. Adding Borders Around the Image
# ─────────────────────────────────────────────
# All borders add 100 pixels on each side (top, bottom, left, right)

# REPLICATE: extends edge pixel values outward (repeats the border row/col)
replicate = cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_REPLICATE)

# CONSTANT: fills border with a solid color — here black (0, 0, 0) in BGR
constant = cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_CONSTANT, value=(0, 0, 0))

# REFLECT: mirrors image content at the border (e.g., abcde → edcba|abcde|edcba)
reflect = cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_REFLECT)

# REFLECT_101: same as REFLECT but excludes the border pixel itself
# (e.g., abcde → dcba|abcde|dcba) — avoids border pixel duplication
reflect_101 = cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_REFLECT_101)

# WRAP: tiles the image — opposite edge pixels fill the border
wrap = cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_WRAP)

# Display all five border variants for visual comparison
cv.imshow("Replicate", replicate)
cv.imshow("Constant", constant)
cv.imshow("Reflect", reflect)
cv.imshow("Reflect_101", reflect_101)
cv.imshow("Wrap", wrap)

# Wait for a key press before exiting
cv.waitKey(0)

# Close all OpenCV display windows
cv.destroyAllWindows()

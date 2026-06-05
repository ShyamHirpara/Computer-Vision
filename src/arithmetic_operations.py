"""
arithmetic_operations.py
------------------------
Demonstrates image blending and bitwise operations using OpenCV.
Goal: Overlay a logo (img2) onto the top-left corner of a background image (img1)
using masking techniques — both at full opacity and reduced opacity.
"""

import cv2 as cv
import numpy as np

# ─────────────────────────────────────────────
# 1. Setup & Image Loading
# ─────────────────────────────────────────────

# Directory where source images are stored
dir_path = "src\\captures\\"

# Load the background image (e.g., a photo of Messi)
img1 = cv.imread(dir_path + 'messi5.jpg')

# Load the logo image to be overlaid (white background logo)
img2 = cv.imread(dir_path + 'opencv-logo-white.png')

# Ensure both images were loaded successfully; abort if not
assert img1 is not None, "file could not be read, check with os.path.exists()"
assert img2 is not None, "file could not be read, check with os.path.exists()"

# ─────────────────────────────────────────────
# 2. Define Region of Interest (ROI)
# ─────────────────────────────────────────────

# Get the dimensions of the logo image
rows, cols, channels = img2.shape

# Crop a region from the top-left of img1 that matches the logo's size
# This is the area where the logo will be placed
roi = img1[0:rows, 0:cols]

# ─────────────────────────────────────────────
# 3. Create Binary Mask from the Logo
# ─────────────────────────────────────────────

# Convert the logo to grayscale to prepare for thresholding
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Apply binary threshold: pixels above intensity 10 become white (255), rest become black (0)
# This creates a mask that isolates the logo shape
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)

# Invert the mask: logo area becomes black, background becomes white
# Used to "cut out" the logo-shaped hole from the background ROI
mask_inv = cv.bitwise_not(mask)

# ─────────────────────────────────────────────
# 4. Separate Background & Foreground
# ─────────────────────────────────────────────

# Black-out the logo area in the ROI (background only, no logo pixels)
# Uses inverted mask so only the non-logo region is kept from img1
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

# Extract only the logo pixels from img2 (foreground only)
# Uses original mask so only the logo-shaped region is kept from img2
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

# ─────────────────────────────────────────────
# Intermediate Result Viewers (Uncomment to Debug)
# ─────────────────────────────────────────────
# cv.imshow("mask_inv", mask_inv)   # Shows inverted mask
# cv.imshow("img2gray", img2gray)   # Shows grayscale logo
# cv.imshow("mask", mask)           # Shows binary mask
# cv.imshow("img1_bg", img1_bg)     # Shows background with logo hole
# cv.imshow("img2_fg", img2_fg)     # Shows extracted logo
# cv.waitKey(0)

# ─────────────────────────────────────────────
# 5. Full Opacity Logo Overlay
# ─────────────────────────────────────────────

# Combine the background (with hole) and the logo foreground pixel-by-pixel
# Result: logo placed seamlessly onto the background at full opacity
dst = cv.add(img1_bg, img2_fg)

# Write the combined result back into the original image's top-left ROI
img1[0:rows, 0:cols] = dst

# Display the image with the logo at full opacity
cv.imshow('image', img1)

# ─────────────────────────────────────────────
# 6. Reduced Opacity Logo Overlay (Blended)
# ─────────────────────────────────────────────

# Blend the background and logo with weighted addition:
#   - img1_bg weight = 1.0 (full background)
#   - img2_fg weight = 0.3 (30% logo opacity → semi-transparent effect)
#   - scalar = 0 (no brightness offset)
dst = cv.addWeighted(img1_bg, 1.0, img2_fg, 0.3, 0)

# Write the blended result back into the image's top-left ROI
img1[0:rows, 0:cols] = dst

# Display the image with the logo at reduced (30%) opacity
cv.imshow('image2', img1)

# ─────────────────────────────────────────────
# 7. Cleanup
# ─────────────────────────────────────────────

# Wait indefinitely until the user presses any key
cv.waitKey(0)

# Close all OpenCV display windows
cv.destroyAllWindows()
"""
simple_thresholding.py
-----------------------
Demonstrates the five simple (global) thresholding modes in OpenCV on a live webcam feed.
A single threshold value (127) is applied uniformly across the entire grayscale frame.

Threshold modes covered:
  1. THRESH_BINARY      — pixels > thresh → 255, else → 0
  2. THRESH_BINARY_INV  — pixels > thresh → 0,   else → 255  (inverted binary)
  3. THRESH_TRUNC       — pixels > thresh → thresh, else unchanged
  4. THRESH_TOZERO      — pixels > thresh unchanged, else → 0
  5. THRESH_TOZERO_INV  — pixels > thresh → 0, else unchanged

Press 'q' to quit.
Reference: https://docs.opencv.org/4.13.0/d7/d4d/tutorial_py_thresholding.html
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. Initialize Webcam
# ─────────────────────────────────────────────

# Open default webcam (index 0)
cap = cv.VideoCapture(0)

# ─────────────────────────────────────────────
# 2. Main Processing Loop
# ─────────────────────────────────────────────

while True:
    # Capture one frame; ret=True if successful
    ret, frame = cap.read()

    # Exit if frame cannot be read
    if not ret:
        print("Frame can't be received")
        break

    # Mirror frame horizontally for natural selfie view
    frame = cv.flip(frame, 1)

    # Convert to grayscale — thresholding requires single-channel input
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # ── Thresholding ─────────────────────────────────────────────────────────
    # cv.threshold(src, thresh, maxval, type) → (retval, dst)
    # All modes below use: thresh=127, maxval=255
    # retval is the threshold used (same as thresh for simple thresholding)

    # thresh1: BINARY — pixel > 127 → 255 (white), else → 0 (black)
    # Produces a clean black-and-white segmentation
    ret, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # thresh2: BINARY_INV — pixel > 127 → 0 (black), else → 255 (white)
    # Inverted version of BINARY; dark objects become white foreground
    ret, thresh2 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

    # thresh3: TRUNC — pixel > 127 → capped at 127, else unchanged
    # Clips bright pixels to the threshold, preserving dark pixel intensities
    ret, thresh3 = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)

    # thresh4: TOZERO — pixel > 127 → kept as-is, else → 0
    # Suppresses dark pixels to black; bright pixels retain original intensity
    ret, thresh4 = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO)

    # thresh5: TOZERO_INV — pixel > 127 → 0, else → kept as-is
    # Suppresses bright pixels to black; dark pixels retain original intensity
    ret, thresh5 = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO_INV)

    # ── Display all results ───────────────────────────────────────────────────
    cv.imshow("gray frame", gray)     # Original grayscale input
    cv.imshow("thresh1", thresh1)     # Binary
    cv.imshow("thresh2", thresh2)     # Binary Inverted
    cv.imshow("thresh3", thresh3)     # Truncate
    cv.imshow("thresh4", thresh4)     # To Zero
    cv.imshow("thresh5", thresh5)     # To Zero Inverted

    # Press 'q' to exit the loop
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# ─────────────────────────────────────────────
# 3. Cleanup
# ─────────────────────────────────────────────

# Release webcam resource
cap.release()

# Close all display windows
cv.destroyAllWindows()
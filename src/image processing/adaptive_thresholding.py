"""
adaptive_thresholding.py
-------------------------
Demonstrates adaptive thresholding on a live webcam feed.

Unlike simple (global) thresholding which uses one fixed threshold for the entire image,
adaptive thresholding computes a LOCAL threshold for each pixel based on the intensity
of its surrounding neighbourhood. This makes it far more robust under:
  - Uneven lighting conditions
  - Shadows or gradients across the image

Two adaptive methods are compared:
  1. ADAPTIVE_THRESH_MEAN_C     — threshold = mean of neighbourhood - C
  2. ADAPTIVE_THRESH_GAUSSIAN_C — threshold = gaussian-weighted mean of neighbourhood - C

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

    # Exit if frame cannot be read (e.g., camera disconnected)
    if not ret:
        print("Frame can't be received")
        break

    # Mirror frame horizontally for natural selfie view
    frame = cv.flip(frame, 1)

    # Convert to grayscale — adaptive thresholding requires single-channel input
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # ── Adaptive Thresholding ─────────────────────────────────────────────────
    # cv.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    #
    #   src           : grayscale input image
    #   maxValue      : value assigned to pixels that pass the threshold (255 = white)
    #   adaptiveMethod: how the local threshold is calculated (MEAN_C or GAUSSIAN_C)
    #   thresholdType : THRESH_BINARY or THRESH_BINARY_INV
    #   blockSize     : size of the neighbourhood area (must be odd, e.g. 11)
    #                   larger = smoother result, smaller = finer local detail
    #   C             : constant subtracted from the computed mean/weighted mean
    #                   positive C reduces sensitivity (raises effective threshold)

    # thresh1: MEAN_C — local threshold = average of all pixels in 11×11 block − 2
    # Simple and fast; can be noisy in textured regions
    thresh1 = cv.adaptiveThreshold(gray, 255,
                                   cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY,
                                   blockSize=11, C=2)

    # thresh2: GAUSSIAN_C — local threshold = gaussian-weighted average in 11×11 block − 2
    # Pixels closer to the centre have more influence; produces smoother, less noisy edges
    thresh2 = cv.adaptiveThreshold(gray, 255,
                                   cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY,
                                   blockSize=11, C=2)

    # ── Display results ───────────────────────────────────────────────────────
    cv.imshow("gray frame", gray)    # Original grayscale input
    cv.imshow("thresh1", thresh1)    # Mean adaptive threshold
    cv.imshow("thresh2", thresh2)    # Gaussian adaptive threshold

    # Press 'q' to quit
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
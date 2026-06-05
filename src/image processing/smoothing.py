"""
smoothing.py
-------------
Demonstrates four image smoothing (blurring) techniques on a live webcam feed.
Smoothing reduces image noise and detail — useful as a pre-processing step before
edge detection, object recognition, or other computer vision tasks.

Techniques covered (uncomment to enable):
  1. Averaging         — simple box filter; uniform weight for all neighbours
  2. Gaussian Blur     — gaussian-weighted filter; reduces noise while preserving edges better
  3. Median Blur       — replaces each pixel with the median of its neighbourhood; great for salt-and-pepper noise
  4. Bilateral Filter  — edge-preserving smoothing; blurs colour but keeps sharp edges intact

Press 'q' to quit.
Reference: https://docs.opencv.org/4.13.0/d4/d13/tutorial_py_filtering.html
"""

import cv2 as cv
import numpy as np

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
        print('cant read frame.')
        break

    # Mirror frame horizontally for natural selfie view
    frame = cv.flip(frame, 1)

    # ── 2.1 Averaging (Box Filter) ────────────────────────────────────────────
    # Convolves the image with a kernel where every element has equal weight (1/25).
    # This is the simplest blur — fast but can smear edges significantly.
    #
    # Option A: Manual kernel convolution with filter2D
    #   kernel = np.ones((5,5), np.float32) / 25  # 5×5 kernel, each weight = 1/25
    #   dst = cv.filter2D(frame, -1, kernel)       # -1 = output depth same as input
    #
    # Option B: Shorthand using cv.blur (does the same thing)
    #   blur = cv.blur(frame, ksize=(5,5))         # 5×5 averaging kernel

    # ── 2.2 Gaussian Blur ─────────────────────────────────────────────────────
    # Applies a Gaussian (bell-curve) weighted kernel — pixels closer to the centre
    # have more influence than those at the edges of the kernel.
    # Better than averaging for natural-looking blur and noise reduction.
    #
    # cv.GaussianBlur(src, ksize, sigmaX):
    #   ksize  = (5, 5) — kernel size (must be odd); larger = more blur
    #   sigmaX = 0      — OpenCV auto-calculates sigma from ksize when set to 0
    gblur = cv.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

    # ── 2.3 Median Blur ───────────────────────────────────────────────────────
    # Replaces each pixel with the MEDIAN value of its ksize×ksize neighbourhood.
    # Highly effective at removing salt-and-pepper (impulse) noise.
    # Non-linear filter — preserves edges better than averaging.
    #
    # cv.medianBlur(src, ksize):
    #   ksize = 5 — neighbourhood size (must be odd positive integer)
    # median = cv.medianBlur(frame, ksize=5)

    # ── 2.4 Bilateral Filter ──────────────────────────────────────────────────
    # An edge-preserving smoothing filter — blurs areas of similar colour
    # while keeping sharp boundaries (edges) largely intact.
    # Significantly slower than the other filters but produces best quality.
    #
    # cv.bilateralFilter(src, d, sigmaColor, sigmaSpace):
    #   d          = 9  — diameter of pixel neighbourhood used during filtering
    #   sigmaColor = 75 — range of colours to mix; larger = wider colour range blended
    #   sigmaSpace = 75 — spatial spread; larger = farther pixels influence each other
    # bilateral = cv.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # ── Display Active Filter Results ─────────────────────────────────────────
    # Uncomment the imshow lines that match the filter you have enabled above

    # cv.imshow("Averaged (filter2D)", dst)   # Manual kernel result
    # cv.imshow("Averaged (blur)", blur)       # cv.blur shorthand result
    cv.imshow("Gaussian Blur", gblur)          
    # cv.imshow("Median Blur", median)
    # cv.imshow("Bilateral Filter", bilateral)

    # Press 'q' to exit
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# ─────────────────────────────────────────────
# 3. Cleanup
# ─────────────────────────────────────────────

# Release the webcam back to the OS
cap.release()

# Close all OpenCV display windows
cv.destroyAllWindows()

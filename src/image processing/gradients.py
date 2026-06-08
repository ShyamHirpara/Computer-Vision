# reference : https://docs.opencv.org/4.13.0/d5/d0f/tutorial_py_gradients.html
"""
gradients.py
--------------
Detects intensity gradients (edges) in a live webcam feed using four methods:
  1. Sobel X      — derivative along X-axis (vertical edges)
  2. Sobel Y      — derivative along Y-axis (horizontal edges)
  3. Laplacian    — second-derivative (detects edges in all directions)
  4. Canny        — multi-stage algorithm: noise reduction → intensity gradient → non-maximum suppression → hysteresis thresholding

Press 'q' to quit.
Reference: https://docs.opencv.org/4.13.0/d5/d0f/tutorial_py_gradients.html
"""

import cv2 as cv
import numpy as np

# ─────────────────────────────────────────────
# 1. Initialize Webcam
# ─────────────────────────────────────────────
cap = cv.VideoCapture(0)

# ─────────────────────────────────────────────
# 2. Main Processing Loop
# ─────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't read frame")
        break
    
    # Flip horizontally for natural selfie view
    frame = cv.flip(frame, 1)
    
    # Convert to grayscale — gradient operations work on single-channel images
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # ── 2.1 Sobel X-derivative ───────────────────────────────────────────────
    # Detects vertical edges (changes along X-axis).
    # 
    # cv.Sobel(src, ddepth, dx, dy, ksize):
    #   ddepth = cv.CV_64F — output depth; use 64-bit float to avoid overflow during differentiation
    #   dx     = 1         — derivative order in X direction
    #   dy     = 0         — derivative order in Y direction
    #   ksize  = 5         — Sobel kernel size (odd integer)
    sobelx = cv.Sobel(gray, cv.CV_64F, dx=1, dy=0, ksize=5)
    
    # Convert back to uint8 for display (scale to 0-255 range)
    sobelx = cv.convertScaleAbs(sobelx)

    # ── 2.2 Sobel Y-derivative ───────────────────────────────────────────────
    # Detects horizontal edges (changes along Y-axis).
    # 
    # dx = 0, dy = 1: compute derivative along Y-axis
    sobely = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1, ksize=5)
    sobely = cv.convertScaleAbs(sobely)

    # ── 2.3 Laplacian Operator ───────────────────────────────────────────────
    # Second-derivative operator — detects all edges (ridges and valleys).
    # Sensitive to noise, so usually applied after Gaussian smoothing.
    # 
    # cv.Laplacian(src, ddepth, ksize):
    #   ddepth = cv.CV_64F — output depth
    #   ksize  = 5         — Laplacian kernel size (odd integer)
    laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=5)
    laplacian = cv.convertScaleAbs(laplacian)

    # ── 2.4 Canny Edge Detection ───────────────────────────────────────────────
    # Multi-stage algorithm widely used in computer vision:
    #   1. Noise reduction using Gaussian filter
    #   2. Compute intensity gradients using Sobel
    #   3. Non-maximum suppression (thin edges)
    #   4. Hysteresis thresholding to connect/discard edges
    # 
    # cv.Canny(image, threshold1, threshold2):
    #   threshold1, threshold2 — the two hysteresis thresholds
    #   Edges with gradient > threshold2 are sure edges
    #   Edges with gradient < threshold1 are discarded
    #   Edges between the two are included only if connected to sure edges
    canny = cv.Canny(gray, 100, 200)  # typical thresholds; adjust based on lighting

    # ── Display Results ──────────────────────────────────────────────────────
    cv.imshow("Original", frame)
    cv.imshow("Sobel X (vertical edges)", sobelx)
    cv.imshow("Sobel Y (horizontal edges)", sobely)
    cv.imshow("Laplacian (all edges)", laplacian)
    cv.imshow("Canny Edges", canny)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

# ─────────────────────────────────────────────
# 3. Cleanup
# ─────────────────────────────────────────────
cap.release()
cv.destroyAllWindows()

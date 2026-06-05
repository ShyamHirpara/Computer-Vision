"""
color_space_change.py
----------------------
Captures live webcam video and demonstrates real-time color space conversions:
  - BGR  : Original color feed from the webcam
  - Gray : Grayscale conversion (removes color, keeps luminance)
  - HSV  : Hue-Saturation-Value space (useful for color-based detection)

All three streams are simultaneously displayed and saved as .avi video files.
reference : https://docs.opencv.org/4.13.0/df/d9d/tutorial_py_colorspaces.html
"""

import cv2 as cv
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. Initialize Webcam
# ─────────────────────────────────────────────

# Open the default webcam (index 0); use 1, 2, etc. for external cameras
cap = cv.VideoCapture(0)

# Directory where output video files will be saved
dir_path = "src\\captures\\"

# ─────────────────────────────────────────────
# 2. Configure Video Writers
# ─────────────────────────────────────────────

# XVID is a widely supported MPEG-4 codec for .avi format
fourcc = cv.VideoWriter_fourcc(*'XVID')

# Frame rate of the output videos (frames per second)
fps = 20.0

# Resolution of each saved video frame (width × height)
frame_size = (640, 480)

# Writer for the original BGR (color) video
out_bgr = cv.VideoWriter(os.path.join(dir_path, 'BGR_video.avi'), fourcc, fps, frame_size)

# Writer for the grayscale video — isColor=False tells OpenCV to expect single-channel frames
out_gray = cv.VideoWriter(os.path.join(dir_path, 'gray_video.avi'), fourcc, fps, frame_size, isColor=False)

# Writer for the HSV color space video
out_hsv = cv.VideoWriter(os.path.join(dir_path, 'HSV_video.avi'), fourcc, fps, frame_size)

# ─────────────────────────────────────────────
# 3. Main Capture & Processing Loop
# ─────────────────────────────────────────────

while True:
    # Read one frame from the webcam
    # ret = True if frame was captured successfully, False otherwise
    ret, frame = cap.read()

    # If frame capture failed (e.g., camera disconnected), exit the loop
    if not ret:
        print("frame can't be received")
        break

    # Mirror the frame horizontally (flip code 1) for a natural "selfie" view
    frame = cv.flip(frame, 1)

    # Convert BGR frame to Grayscale — reduces 3 channels to 1 (luminance only)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Convert BGR frame to HSV — separates color (Hue) from brightness (Value)
    # Useful for robust color detection under varying lighting conditions
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # ── Display all three color spaces in real time ──
    cv.imshow('color frame', frame)   # Original BGR feed
    cv.imshow("gray frame", gray)     # Grayscale feed
    cv.imshow("hsv frame", hsv)       # HSV feed

    # ── Write current frame to each respective output video file ──
    out_bgr.write(frame)   # Save BGR frame
    out_gray.write(gray)   # Save grayscale frame (single-channel)
    out_hsv.write(hsv)     # Save HSV frame

    # Wait 1 ms for a key press; press 'q' to quit the loop
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# ─────────────────────────────────────────────
# 4. Cleanup — Release Resources
# ─────────────────────────────────────────────

# Release the webcam so it can be used by other applications
cap.release()

# Finalize and close all three output video files
out_bgr.release()
out_gray.release()
out_hsv.release()

# Close all OpenCV display windows
cv.destroyAllWindows()

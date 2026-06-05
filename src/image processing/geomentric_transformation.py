"""
geomentric_transformation.py
-----------------------------
Demonstrates five geometric transformations on a live webcam feed using OpenCV:
  1. Scaling    — resize the frame up or down
  2. Translation — shift the frame along X and Y axes
  3. Rotation   — rotate the frame around its center
  4. Affine     — warp using 3-point correspondence (preserves parallel lines)
  5. Perspective — warp using 4-point correspondence (simulates 3D tilt/perspective)

Press 'q' to quit.
Reference: https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html
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
    # Capture one frame from the webcam
    # ret = True if capture was successful
    ret, frame = cap.read()

    # Exit loop if frame can't be read (e.g., camera disconnected)
    if not ret:
        print('could not read frame')
        break

    # Flip horizontally (code 1) for natural mirror/selfie view
    frame = cv.flip(frame, 1)

    # Extract frame dimensions for use in transformation matrices
    rows, cols, ch = frame.shape

    # Show the original (unmodified) frame
    cv.imshow("Image", frame)

    # ── 2.1 Scaling ──────────────────────────────────────────────────────────
    # Resize the frame by scale factors fx=1.25 (width) and fy=1.25 (height)
    # INTER_CUBIC uses bicubic interpolation — best quality but slower than INTER_LINEAR
    # None for dsize means the output size is computed from the scale factors
    res = cv.resize(frame, None, fx=1.25, fy=1.25, interpolation=cv.INTER_CUBIC)
    cv.imshow("Resized", res)

    # ── 2.2 Translation ──────────────────────────────────────────────────────
    # Translation shifts the image by (tx, ty) pixels
    # Transformation matrix format:
    #   M = [[1, 0, tx],   ← shifts right by tx pixels
    #        [0, 1, ty]]   ← shifts down by ty pixels
    # Here: shift right 100px, shift down 50px
    M = np.float32([[1, 0, 100],
                    [0, 1, 50]])
    # Apply the translation; output size = (cols, rows) to keep original dimensions
    dst = cv.warpAffine(frame, M, (cols, rows))
    cv.imshow("Translated", dst)

    # ── 2.3 Rotation ─────────────────────────────────────────────────────────
    # Standard rotation matrix:
    #   M = [[ cos θ, -sin θ],
    #        [ sin θ,  cos θ]]
    #
    # OpenCV extends this with a configurable center and scale factor:
    #   M = [[α·cos θ, -β·sin θ, tx],
    #        [β·sin θ,  α·cos θ, ty]]
    # where α = scale·cos θ, β = scale·sin θ, and (tx, ty) adjust for center.
    #
    # cv.getRotationMatrix2D(center, angle, scale):
    #   center = (cols/2, rows/2) → rotate around the image center
    #   angle  = 90               → 90 degrees counter-clockwise
    #   scale  = 1                → no zoom (scale factor = 1)
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv.warpAffine(frame, M, (cols, rows))
    cv.imshow("Rotated", dst)

    # ── 2.4 Affine Transformation ─────────────────────────────────────────────
    # Affine transformation maps 3 source points → 3 destination points.
    # Properties: parallel lines stay parallel, shapes are sheared/skewed/rotated.
    # cv.getAffineTransform(src_pts, dst_pts) computes the 2×3 matrix M.
    #
    # pts1: 3 control points in the source frame
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    # pts2: where those 3 points should map to in the output
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(frame, M, (cols, rows))
    cv.imshow("Affine Transformation", dst)

    # ── 2.5 Perspective Transformation ───────────────────────────────────────
    # Perspective transformation maps 4 source points → 4 destination points.
    # Unlike affine, parallel lines do NOT remain parallel — simulates a 3D viewpoint tilt.
    # cv.getPerspectiveTransform(src_pts, dst_pts) computes a 3×3 homography matrix M.
    # cv.warpPerspective applies the full perspective warp.
    #
    # pts1: 4 control points in the source frame (must be in consistent order)
    pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
    # pts2: where those 4 points map to in the output
    pts2 = np.float32([[10, 100], [200, 50], [100, 250], [250, 250]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(frame, M, (cols, rows))
    cv.imshow("Perspective Transformation", dst)

    # ─────────────────────────────────────────────
    # 3. Key Handler — press 'q' to quit
    # ─────────────────────────────────────────────
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# ─────────────────────────────────────────────
# 4. Cleanup
# ─────────────────────────────────────────────

# Release the webcam back to the OS
cap.release()

# Close all OpenCV windows
cv.destroyAllWindows()

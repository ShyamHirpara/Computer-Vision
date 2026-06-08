"""
box_detector.py  —  White Box / Cube Detection using OpenCV
============================================================
Detects white rectangular objects on a contrasting surface using
classical image processing (HSV thresholding + contour filtering).

Modes: "image" | "camera" | "tuner"  (set MODE at the bottom)
Keys:  q = quit | s = save | t = toggle mask (camera mode)

Dependencies: pip install opencv-python numpy
Author: Shyam Hirpara  |  Date: 2026-06-08
"""

import cv2 as cv
import numpy as np


# ── TUNABLE PARAMETERS ────────────────────────────────────────────────────────
# Calibrate HSV values for your lighting using MODE = "tuner"
HSV_LOWER_WHITE = np.array([40,  0,  150])   # [H_min, S_min, V_min]
HSV_UPPER_WHITE = np.array([180, 85, 255])   # [H_max, S_max, V_max]

MIN_AREA      = 900       # min blob area (pixels²) — rejects noise
MAX_AREA      = 40_000    # max blob area (pixels²) — rejects merged blobs
MIN_ASPECT    = 0.40      # min width/height ratio
MAX_ASPECT    = 2.20      # max width/height ratio
MIN_RECT_FILL = 0.60      # min contour/bbox area ratio — rejects irregular shapes

ROI_MARGIN_X  = 50        # ignore detections within this many px of left/right edge
ROI_MARGIN_Y  = 30        # ignore detections within this many px of top/bottom edge

MORPH_KERNEL  = 7         # kernel size for OPEN/CLOSE morphological ops
SEVER_KERNEL  = 5         # kernel size for cable-severing erode/dilate
BLUR_KERNEL   = 5         # Gaussian blur kernel (must be odd)

BOX_COLOR     = (0, 0, 255)      # BGR: red bounding box
TEXT_COLOR    = (255, 255, 255)  # BGR: white labels
FONT          = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.5
FONT_THICKNESS = 1
# ──────────────────────────────────────────────────────────────────────────────


def preprocess(frame):
    """
    Convert a BGR frame into a binary mask of white object regions.

    Args:
        frame (np.ndarray): Input BGR image.

    Returns:
        mask (np.ndarray): Binary (0/255) grayscale mask, same H×W as frame.
    """
    blurred = cv.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
    hsv     = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    mask    = cv.inRange(hsv, HSV_LOWER_WHITE, HSV_UPPER_WHITE)

    k_main  = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    k_sever = np.ones((SEVER_KERNEL, SEVER_KERNEL), np.uint8)

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k_main)   # remove noise
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k_main)   # fill holes

    # Cable-severing pass: thin cables disappear, thick cubes survive and recover
    mask = cv.erode(mask,  k_sever, iterations=6)
    mask = cv.dilate(mask, k_sever, iterations=8)

    return mask


def is_box_shaped(cnt, x, y, w, h, frame_shape):
    """
    Return True if the contour passes all 5 box-shape validation tests:
      1. Area within [MIN_AREA, MAX_AREA]
      2. Aspect ratio within [MIN_ASPECT, MAX_ASPECT]
      3. Fill ratio (contour/bbox area) >= MIN_RECT_FILL
      4. Polygon approximation has 3–6 sides
      5. Blob center is inside the ROI (not too close to frame edges)

    Args:
        cnt         : OpenCV contour.
        x, y, w, h  : Bounding rectangle of the contour.
        frame_shape : Shape of the original frame (height, width, channels).

    Returns:
        bool: True if all tests pass.
    """
    area      = cv.contourArea(cnt)
    bbox_area = w * h
    if bbox_area == 0:
        return False

    aspect     = w / float(h)
    fill_ratio = area / float(bbox_area)

    epsilon = 0.06 * cv.arcLength(cnt, True)
    approx  = cv.approxPolyDP(cnt, epsilon, True)
    sides   = len(approx)

    cx, cy  = x + w // 2, y + h // 2
    fh, fw  = frame_shape[:2]
    in_roi  = (
        cx > ROI_MARGIN_X        and
        cy > ROI_MARGIN_Y        and
        cx < (fw - ROI_MARGIN_X) and
        cy < (fh - ROI_MARGIN_Y)
    )

    return (
        MIN_AREA   < area   < MAX_AREA  and
        MIN_ASPECT < aspect < MAX_ASPECT and
        fill_ratio > MIN_RECT_FILL       and
        3 <= sides <= 6                  and
        in_roi
    )


def detect_boxes(frame):
    """
    Run the full detection pipeline on one frame and return annotated results.

    Args:
        frame (np.ndarray): Input BGR image.

    Returns:
        output (np.ndarray): Annotated frame with bounding boxes drawn.
        mask   (np.ndarray): Binary mask from preprocess().
        boxes  (list[dict]): Detected boxes — keys: 'pt1', 'pt2', 'center', 'area'.
    """
    mask   = preprocess(frame)
    output = frame.copy()
    boxes  = []

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if not is_box_shaped(cnt, x, y, w, h, frame.shape):
            continue

        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cx  = x + w // 2
        cy  = y + h // 2

        cv.rectangle(output, pt1, pt2, BOX_COLOR, 2)
        cv.putText(output, f"({x},{y})", pt1,
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
        cv.putText(output, f"({x+w},{y+h})", (x + w - 60, y + h + 14),
                   FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)
        cv.circle(output, (cx, cy), 4, BOX_COLOR, -1)

        boxes.append({"pt1": pt1, "pt2": pt2, "center": (cx, cy),
                      "area": cv.contourArea(cnt)})

    cv.putText(output, f"Boxes detected: {len(boxes)}", (10, 28),
               FONT, 0.8, (0, 255, 0), 2, cv.LINE_AA)

    return output, mask, boxes


# ── RUN MODES ─────────────────────────────────────────────────────────────────

def run_on_image(path: str):
    """
    Detect boxes on a saved image. Shows mask (left) + result (right) side by side.

    Keys: q = close | s = save detection_output.jpg + detection_mask.jpg

    Args:
        path (str): Path to the input image (e.g. 'test.jpg').
    """
    img = cv.imread(path)
    if img is None:
        print(f"[ERROR] Could not read image: {path}")
        return

    result, mask, boxes = detect_boxes(img)

    print(f"[INFO] Detected {len(boxes)} box(es)")
    for i, b in enumerate(boxes):
        print(f"  Box {i+1}: {b['pt1']} -> {b['pt2']}  "
              f"centre={b['center']}  area={b['area']:.0f}")

    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    combined = np.hstack([mask_bgr, result])
    cv.namedWindow("Box Detector -- [q] quit  [s] save", cv.WINDOW_NORMAL)
    cv.imshow("Box Detector -- [q] quit  [s] save", combined)

    if cv.waitKey(0) == ord('s'):
        cv.imwrite('detection_output.jpg', result)
        cv.imwrite('detection_mask.jpg',   mask)
        print("[INFO] Saved detection_output.jpg and detection_mask.jpg")

    cv.destroyAllWindows()


def run_on_camera(cam_index: int = 0):
    """
    Run real-time box detection from a webcam feed.

    Keys: q = quit | s = save snapshot | t = toggle result/mask view

    Args:
        cam_index (int): Camera device index (0 = default webcam).
    """
    cap = cv.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_index}")
        return

    print("[INFO] Camera started. Keys: [q] quit  [s] save  [t] toggle mask")
    show_mask = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame.")
            break

        result, mask, boxes = detect_boxes(frame)

        display = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) if show_mask else result
        cv.imshow("Box Detector -- Live", display)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv.imwrite('snapshot.jpg', frame)
            cv.imwrite('snapshot_result.jpg', result)
            print("[INFO] Saved snapshot.jpg and snapshot_result.jpg")
        elif key == ord('t'):
            show_mask = not show_mask
            print(f"[INFO] Display mode -> {'MASK' if show_mask else 'RESULT'}")

    cap.release()
    cv.destroyAllWindows()


def run_hsv_tuner(cam_index: int = 0):
    """
    Interactive HSV calibration tool with live trackbars.

    Drag sliders until only the target objects appear white in the mask.
    Press q to quit — tuned values are printed to the terminal.
    Copy those values into HSV_LOWER_WHITE / HSV_UPPER_WHITE above.

    Args:
        cam_index (int): Camera device index (0 = default webcam).
    """
    cap = cv.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_index}")
        return

    cv.namedWindow("HSV Tuner", cv.WINDOW_NORMAL)
    cv.createTrackbar("H Low",  "HSV Tuner",   0, 180, lambda v: None)
    cv.createTrackbar("H High", "HSV Tuner", 180, 180, lambda v: None)
    cv.createTrackbar("S Low",  "HSV Tuner",   0, 255, lambda v: None)
    cv.createTrackbar("S High", "HSV Tuner",  60, 255, lambda v: None)
    cv.createTrackbar("V Low",  "HSV Tuner", 160, 255, lambda v: None)
    cv.createTrackbar("V High", "HSV Tuner", 255, 255, lambda v: None)

    print("[INFO] HSV Tuner running. Drag sliders. Press [q] to quit and print values.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hl = cv.getTrackbarPos("H Low",  "HSV Tuner")
        hh = cv.getTrackbarPos("H High", "HSV Tuner")
        sl = cv.getTrackbarPos("S Low",  "HSV Tuner")
        sh = cv.getTrackbarPos("S High", "HSV Tuner")
        vl = cv.getTrackbarPos("V Low",  "HSV Tuner")
        vh = cv.getTrackbarPos("V High", "HSV Tuner")

        hsv  = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))

        combined = np.hstack([frame, cv.cvtColor(mask, cv.COLOR_GRAY2BGR)])
        cv.imshow("HSV Tuner", combined)

        if cv.waitKey(1) & 0xFF == ord('q'):
            print(f"\n[TUNED VALUES — copy these into your script]")
            print(f"HSV_LOWER_WHITE = np.array([{hl}, {sl}, {vl}])")
            print(f"HSV_UPPER_WHITE = np.array([{hh}, {sh}, {vh}])")
            break

    cap.release()
    cv.destroyAllWindows()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODE = "camera"   # "image" | "camera" | "tuner"
    # MODE = "image"
    # MODE = "tuner"

    if MODE == "image":
        run_on_image("test.jpg")
    elif MODE == "camera":
        run_on_camera(cam_index=0)
    elif MODE == "tuner":
        run_hsv_tuner(cam_index=0)
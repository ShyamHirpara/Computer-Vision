"""
distance_estimator.py  —  Live Multi-Object solvePnP Distance Estimation
==========================================================================
Detects and classifies white objects (Small Cube, Large Cube, Cylinder)
using object_detector.py, then estimates each object's distance from the
Lenovo 300 webcam using the Perspective-n-Point (solvePnP) algorithm.

MATH (Pinhole Camera Model):
-----------------------------
    s × [u, v, 1]T  =  K × [R | t] × [X, Y, Z, 1]T

Where K (intrinsic matrix) is loaded from camera_calibration.npz.
cv.solvePnP() solves for t (translation vector). t[2] = Z = depth (cm).

Per-object solvePnP:
  Each detected + classified object uses its own real-world front-face
  dimensions (from config.OBJECTS) to build 4 object points, then matches
  them to the 4 bounding-box pixel corners as image points.

MODES:
------
  camera  — live webcam stream (default)
  image   — single saved image

Keys:
  q → quit
  s → save snapshot
  m → toggle mask view
  a → toggle 3D axes
  p → toggle pose (pitch/yaw/roll) text
  c → print current detections to terminal

Author: Shyam Hirpara  |  Date: 2026-06-09
"""

import cv2 as cv
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from object_detector import detect_objects, preprocess


# ── LOAD CALIBRATION ──────────────────────────────────────────────────────────

def load_calibration() -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera matrix K and distortion coefficients from config.CALIB_FILE.

    Returns:
        (camera_matrix, dist_coeffs): Both are np.ndarray.

    Raises:
        SystemExit: If the calibration file does not exist.
    """
    path = config.CALIB_FILE
    if not os.path.exists(path):
        print(f"[ERROR] Calibration file not found: {path}")
        print("  Run camera_calibration.py first.")
        sys.exit(1)

    data = np.load(path)
    K    = data["camera_matrix"]
    dist = data["dist_coeffs"]
    err  = float(data.get("reprojection_error", -1))

    print(f"[INFO] Calibration loaded from '{path}'")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    if err >= 0:
        quality = "GOOD" if err < 0.5 else ("ACCEPTABLE" if err < 1.0 else "POOR")
        print(f"  Reprojection error: {err:.4f} px  ({quality})")

    return K, dist


# ── GEOMETRY HELPERS ──────────────────────────────────────────────────────────

def build_object_points(w_cm: float, h_cm: float) -> np.ndarray:
    """
    Build 4 coplanar 3D front-face corners of an object (Z=0 plane).

    The origin is at the face centre (top-left clockwise order):
        (-w/2, -h/2, 0)  →  (w/2, -h/2, 0)
              ↓                      ↓
        (-w/2,  h/2, 0)  →  (w/2,  h/2, 0)

    Args:
        w_cm: Front face width in centimetres.
        h_cm: Front face height in centimetres.

    Returns:
        np.ndarray: Shape (4, 3) float32.
    """
    hw, hh = w_cm / 2.0, h_cm / 2.0
    return np.array([
        [-hw, -hh, 0.0],
        [ hw, -hh, 0.0],
        [ hw,  hh, 0.0],
        [-hw,  hh, 0.0],
    ], dtype=np.float32)


def get_image_corners(x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Extract the 4 bounding-box corners as solvePnP image points (clockwise).

    Returns:
        np.ndarray: Shape (4, 1, 2) float32.
    """
    return np.array([
        [[float(x),     float(y)    ]],
        [[float(x + w), float(y)    ]],
        [[float(x + w), float(y + h)]],
        [[float(x),     float(y + h)]],
    ], dtype=np.float32)


def triangle_similarity_distance(pixel_w: int, pixel_h: int,
                                 real_w_cm: float, real_h_cm: float,
                                 K: np.ndarray,
                                 obj_cx: int, obj_cy: int) -> float:
    """
    Estimate object distance using the Triangle Similarity (focal length) formula.

    This method is robust across the entire camera frame, including off-center
    positions where solvePnP with axis-aligned bounding box corners is inaccurate.

    Formula:
        d_w = (real_width_cm  x focal_length_x) / effective_pixel_width
        d_h = (real_height_cm x focal_length_y) / effective_pixel_height
        d   = (d_w + d_h) / 2

    Tilt compensation (for near-square objects):
        When a cube is viewed at an angle, its bounding box width is inflated by
        the visible side face.  For objects whose real w ≈ h (square face), we
        substitute min(pixel_w, pixel_h) for both axes, because the shorter bbox
        dimension is far less distorted by horizontal tilt than the wider one.

    Ray-angle → Euclidean correction:
        Triangle Similarity gives Z-depth (distance along the optical axis).
        We convert to the true physical (Euclidean) distance by multiplying by
        the secant of the ray angle:
            d_euclidean = d_z × √(1 + (Δx/fx)² + (Δy/fy)²)
        where Δx, Δy are pixel offsets from the principal point.

    Args:
        pixel_w   : Bounding box width in pixels.
        pixel_h   : Bounding box height in pixels.
        real_w_cm : Known real-world object width in cm.
        real_h_cm : Known real-world object height in cm.
        K         : 3x3 camera intrinsic matrix.
        obj_cx    : Bounding box centre x in pixels.
        obj_cy    : Bounding box centre y in pixels.

    Returns:
        float: Estimated Euclidean distance in centimetres.
    """
    fx     = K[0, 0]
    fy     = K[1, 1]
    cx_cam = K[0, 2]
    cy_cam = K[1, 2]

    # ── Tilt compensation ────────────────────────────────────────────────────
    # For near-square objects, use min(pixel_w, pixel_h) as the effective
    # dimension so that the wider side-face (visible when tilted) does not
    # shrink the distance estimate.
    if abs(real_w_cm - real_h_cm) < 1.0:   # roughly square face (e.g. cube)
        eff_px_w = min(pixel_w, pixel_h)
        eff_px_h = min(pixel_w, pixel_h)
    else:
        eff_px_w = pixel_w
        eff_px_h = pixel_h

    d_w = (real_w_cm * fx) / max(eff_px_w, 1)
    d_h = (real_h_cm * fy) / max(eff_px_h, 1)

    # Z-depth along the optical axis
    z_depth = (d_w + d_h) / 2.0

    # ── Ray-angle → Euclidean correction ────────────────────────────────────
    # Off-centre objects have a longer Euclidean distance than their Z-depth.
    ray_x = (obj_cx - cx_cam) / fx
    ray_y = (obj_cy - cy_cam) / fy
    euclidean_dist = z_depth * np.sqrt(1.0 + ray_x ** 2 + ray_y ** 2)

    return euclidean_dist


def rodrigues_to_euler(rvec) -> tuple[float, float, float]:
    """
    Convert a solvePnP rotation vector to Euler angles (pitch, yaw, roll) in degrees.

    Args:
        rvec: Rotation vector (3×1 array).

    Returns:
        (pitch, yaw, roll) in degrees.
    """
    R, _ = cv.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = 0.0
    return pitch, yaw, roll


def draw_3d_axes(frame, rvec, tvec, K, dist, cx: int, cy: int) -> None:
    """
    Draw XYZ coordinate axes on the detected object face.

    X = red arrow, Y = green arrow, Z = blue arrow (toward camera).

    Args:
        frame     : Image to draw on (modified in-place).
        rvec, tvec: Pose from solvePnP.
        K, dist   : Camera intrinsics.
        cx, cy    : Pixel centre of the object.
    """
    L = config.AXIS_LENGTH_CM
    pts3d = np.float32([
        [0, 0,  0],   # origin
        [L, 0,  0],   # X
        [0, L,  0],   # Y
        [0, 0, -L],   # Z (toward camera)
    ])
    proj, _ = cv.projectPoints(pts3d, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    o  = (int(proj[0][0]), int(proj[0][1]))
    px = (int(proj[1][0]), int(proj[1][1]))
    py = (int(proj[2][0]), int(proj[2][1]))
    pz = (int(proj[3][0]), int(proj[3][1]))

    cv.arrowedLine(frame, o, px, (0,   0, 255), 2, tipLength=0.3)   # X red
    cv.arrowedLine(frame, o, py, (0, 255,   0), 2, tipLength=0.3)   # Y green
    cv.arrowedLine(frame, o, pz, (255, 0,   0), 2, tipLength=0.3)   # Z blue
    cv.putText(frame, "X", (px[0]+4, px[1]), cv.FONT_HERSHEY_SIMPLEX, 0.38,
               (0,   0, 255), 1, cv.LINE_AA)
    cv.putText(frame, "Y", (py[0]+4, py[1]), cv.FONT_HERSHEY_SIMPLEX, 0.38,
               (0, 255,   0), 1, cv.LINE_AA)
    cv.putText(frame, "Z", (pz[0]+4, pz[1]), cv.FONT_HERSHEY_SIMPLEX, 0.38,
               (255,   0,   0), 1, cv.LINE_AA)


# ── HUD DRAWING HELPERS ──────────────────────────────────────────────────────────

def _draw_corner_bracket(img, x1, y1, x2, y2, color, thickness=2, arm=18):
    """
    Draw corner-bracket style bounding box (4 L-shaped corners only).
    Much cleaner than a full rectangle at high object densities.
    """
    for px, py, sx, sy in [
        (x1, y1, +1, +1), (x2, y1, -1, +1),
        (x1, y2, +1, -1), (x2, y2, -1, -1),
    ]:
        cv.line(img, (px, py), (px + sx * arm, py),            color, thickness, cv.LINE_AA)
        cv.line(img, (px, py), (px,            py + sy * arm), color, thickness, cv.LINE_AA)


def _draw_pill_badge(img, cx, top_y, text, bg_color, text_color=(0, 0, 0),
                     font_scale=0.58, thickness=2):
    """
    Draw a rounded-rectangle (pill) badge centred above top_y.
    Returns the y-coordinate of the badge top edge.
    """
    font = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 10, 5
    bx1 = cx - tw // 2 - pad_x
    bx2 = cx + tw // 2 + pad_x
    by1 = top_y - th - pad_y * 2 - 6
    by2 = top_y - 6
    # filled rounded rect via two overlapping rectangles + circles
    cv.rectangle(img, (bx1 + 8, by1), (bx2 - 8, by2), bg_color, -1)
    cv.rectangle(img, (bx1, by1 + 8), (bx2, by2 - 8), bg_color, -1)
    for cx_, cy_ in [(bx1+8, by1+8), (bx2-8, by1+8), (bx1+8, by2-8), (bx2-8, by2-8)]:
        cv.circle(img, (cx_, cy_), 8, bg_color, -1)
    tx = cx - tw // 2
    ty = by2 - pad_y
    cv.putText(img, text, (tx, ty), font, font_scale, text_color, thickness, cv.LINE_AA)
    return by1


def _draw_id_chip(img, x, y, obj_id, color):
    """
    Draw a small numbered chip in the top-left corner of the bounding box.
    """
    text  = str(obj_id)
    font  = cv.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv.getTextSize(text, font, 0.45, 2)
    px, py = x + 4, y + th + 4
    cv.rectangle(img, (x + 2, y + 2), (x + tw + 8, y + th + 8), color, -1)
    cv.putText(img, text, (px, py), font, 0.45, (0, 0, 0), 2, cv.LINE_AA)


def draw_object_hud(output, det, obj_id: int,
                    show_axes: bool, show_pose: bool,
                    K, dist_coeffs) -> None:
    """
    Render the full per-object HUD overlay onto *output* (in-place).

    Elements drawn:
      • Corner-bracket bounding box (coloured, 2 px)
      • Pill-shaped distance badge above the box (green/orange by proximity)
      • Numbered object ID chip in top-left corner
      • Object label below the box
      • Pixel size text inside the box (small, grey)
      • Pose P/Y/R text inside the box (below centre) — only when show_pose
      • 3D XYZ axes — only when show_axes

    Args:
        output      : Annotated BGR frame (modified in-place).
        det         : Detection dict from detect_objects() + distance/rvec/tvec.
        obj_id      : 1-based object index for the ID chip.
        show_axes   : Whether to draw 3D axes.
        show_pose   : Whether to overlay pitch/yaw/roll text.
        K           : Camera matrix.
        dist_coeffs : Distortion coefficients.
    """
    x, y   = det["pt1"]
    x2, y2 = det["pt2"]
    w      = det["pixel_w"]
    h      = det["pixel_h"]
    cx, cy = det["center"]
    label  = det["label"]
    color  = det["color_bgr"]
    dist   = det.get("distance_cm")
    rvec   = det.get("rvec")
    tvec   = det.get("tvec")

    # ── Corner-bracket bounding box ──────────────────────────────────────
    _draw_corner_bracket(output, x, y, x2, y2, color, thickness=2, arm=20)

    # ── Distance pill badge ────────────────────────────────────────────
    if dist is not None:
        dist_m   = dist / 100.0
        badge_bg = (0, 100, 255) if dist < config.CLOSE_DIST_CM else (30, 200, 30)
        badge_tx = (255, 255, 255)
        badge_str = f"{dist:.1f} cm  |  {dist_m:.2f} m"
    else:
        badge_bg  = (60, 60, 60)
        badge_tx  = (200, 200, 200)
        badge_str = "no distance"
    _draw_pill_badge(output, cx, y, badge_str, badge_bg, badge_tx)

    # ── Object ID chip (top-left corner) ─────────────────────────────
    _draw_id_chip(output, x, y, obj_id, color)

    # ── Object label below the box ──────────────────────────────────
    cv.putText(output, label, (x + 2, y2 + 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv.LINE_AA)

    # ── Pixel size (small grey, centre of box) ───────────────────────
    cv.putText(output, f"{w}×{h}px", (cx - 22, cy - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 160), 1, cv.LINE_AA)

    # ── Pose text inside box (below centre) ────────────────────────
    if show_pose and rvec is not None:
        pitch, yaw, roll = rodrigues_to_euler(rvec)
        pose_str = f"P{pitch:+.0f}° Y{yaw:+.0f}° R{roll:+.0f}°"
        cv.putText(output, pose_str, (cx - 30, cy + 14),
                   cv.FONT_HERSHEY_SIMPLEX, 0.34, (170, 170, 255), 1, cv.LINE_AA)

    # ── 3D axes ───────────────────────────────────────────────────
    if show_axes and rvec is not None and dist is not None:
        draw_3d_axes(output, rvec, tvec, K, dist_coeffs, cx, cy)


# ── MAIN PROCESSING ───────────────────────────────────────────────────────────

def process_frame(frame: np.ndarray,
                  K: np.ndarray,
                  dist: np.ndarray,
                  show_axes: bool = True,
                  show_pose: bool = True) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Run detection, classification, and distance estimation on one frame.

    Distance method (two-method split for accuracy):
      - DISTANCE : Triangle Similarity formula  (accurate everywhere in frame)
      - POSE     : solvePnP(IPPE) (used only for 3D axes / orientation overlay)

    Why not use solvePnP for distance?
      solvePnP with axis-aligned bounding-box corners works well when the
      object is near the image centre.  Off-centre, the axis-aligned bbox does
      not match the skewed projection of the 3D corners, so solvePnP under-
      estimates depth (reads object as closer than it really is).  Triangle
      Similarity uses pixel_width / focal_length which is accurate everywhere.

    Args:
        frame     : Input BGR frame.
        K         : 3x3 camera matrix.
        dist      : Distortion coefficients.
        show_axes : Draw 3D XYZ axes on each object.
        show_pose : Overlay pitch/yaw/roll text.

    Returns:
        output     (np.ndarray)  : Annotated frame.
        mask       (np.ndarray)  : Binary detection mask.
        detections (list[dict])  : Per-object results including distance_cm.
    """
    frame_ud   = cv.undistort(frame, K, dist)
    output     = frame_ud.copy()
    detections = detect_objects(frame_ud)
    mask       = preprocess(frame_ud)

    for i, det in enumerate(detections):
        x, y   = det["pt1"]
        x2, y2 = det["pt2"]
        w      = det["pixel_w"]
        h      = det["pixel_h"]
        cx, cy = det["center"]

        # ── Distance: Triangle Similarity (robust off-centre) ────────────
        distance_cm = None
        rvec = tvec = None

        if det["w_cm"] is not None and det["h_cm"] is not None:
            distance_cm = triangle_similarity_distance(
                w, h, det["w_cm"], det["h_cm"], K, cx, cy
            )
            # Clamp to physically plausible range
            if not (0 < distance_cm < 500):
                distance_cm = None

            # ── Pose only: solvePnP for rvec (axes / rotation) ───────────
            # We use solvePnP ITERATIVE with the triangle-similarity distance
            # as an initial tvec guess, giving a much better rotation estimate.
            obj_pts  = build_object_points(det["w_cm"], det["h_cm"])
            img_pts  = get_image_corners(x, y, w, h)

            # Build initial guess tvec from triangle similarity result
            init_rvec = np.zeros((3, 1), dtype=np.float64)   # neutral rotation
            init_tvec = np.array([[0.0], [0.0], [distance_cm if distance_cm else 30.0]], dtype=np.float64)
            ok, rvec, tvec = cv.solvePnP(
                obj_pts, img_pts, K, dist,
                rvec=init_rvec, tvec=init_tvec,
                useExtrinsicGuess=(distance_cm is not None),
                flags=cv.SOLVEPNP_ITERATIVE
            )
            if not ok:
                rvec = tvec = None

        det["distance_cm"] = distance_cm
        det["rvec"]        = rvec
        det["tvec"]        = tvec

        # ── Draw full HUD for this object ────────────────────────────────────
        draw_object_hud(output, det, i + 1, show_axes, show_pose, K, dist)

    # ── Status panel (dark bottom strip) ─────────────────────────────────────
    n_total = len(detections)
    n_known = sum(1 for d in detections if d.get("distance_cm") is not None)
    fh = output.shape[0]
    cv.rectangle(output, (0, fh - 32), (output.shape[1], fh), (20, 20, 20), -1)
    status = (f"  Objects: {n_total}  ({n_known} measured)   "
              f"[m] mask   [a] axes   [p] pose   [s] save   [q] quit")
    cv.putText(output, status, (6, fh - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv.LINE_AA)

    return output, mask, detections


# ── RUN MODES ─────────────────────────────────────────────────────────────────

def run_camera(cam_index: int = config.CAMERA_INDEX):
    """
    Live webcam distance estimation.

    Args:
        cam_index (int): Camera device index.
    """
    K, dist = load_calibration()

    cap = cv.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {cam_index}")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH,  config.FRAME_W)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    cap.set(cv.CAP_PROP_FPS,config.FPS)

    print(f"\n[INFO] Distance estimator started. Camera {cam_index} @ "
          f"{int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}")
    print("       Keys: [q]=quit  [s]=save  [m]=mask  [a]=axes  [p]=pose  [c]=print")

    show_axes = True
    show_pose = True
    show_mask = False
    prev_time = time.time()

    WIN = "Lenovo 300 Distance Estimator"
    cv.namedWindow(WIN, cv.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame.")
            break

        result, mask, dets = process_frame(frame, K, dist, show_axes, show_pose)

        # FPS chip (top-right dark badge)
        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        fps_text = f"FPS {fps:.0f}"
        (fw_, fh_), _ = cv.getTextSize(fps_text, cv.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        fw2 = result.shape[1]
        cv.rectangle(result, (fw2 - fw_ - 16, 6), (fw2 - 4, fh_ + 14), (20, 20, 20), -1)
        cv.putText(result, fps_text, (fw2 - fw_ - 8, fh_ + 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv.LINE_AA)

        display = cv.cvtColor(mask, cv.COLOR_GRAY2BGR) if show_mask else result
        cv.imshow(WIN, display)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv.imwrite(f"snapshot_{ts}.jpg", result)
            print(f"[INFO] Saved snapshot_{ts}.jpg")
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('a'):
            show_axes = not show_axes
        elif key == ord('p'):
            show_pose = not show_pose
        elif key == ord('c'):
            print(f"\n[DETECTIONS]  {len(dets)} object(s) in frame:")
            for i, d in enumerate(dets):
                dist_str = (f"{d['distance_cm']:.2f} cm"
                            if d.get("distance_cm") is not None else "n/a")
                print(f"  {i+1}. {d['label']:12s}  "
                      f"dist={dist_str:>10}  "
                      f"aspect={d['aspect']:.2f}  fill={d['fill']:.2f}")

    cap.release()
    cv.destroyAllWindows()


def run_image(path: str):
    """
    Run distance estimation on a saved image file.

    Args:
        path (str): Path to input image.
    """
    K, dist = load_calibration()

    img = cv.imread(path)
    if img is None:
        print(f"[ERROR] Cannot read image: {path}")
        return

    result, mask, dets = process_frame(img, K, dist)

    print(f"\n[INFO] Detected {len(dets)} object(s):")
    for i, d in enumerate(dets):
        dist_str = f"{d['distance_cm']:.2f} cm" if d.get("distance_cm") is not None else "n/a"
        print(f"  {i+1}. {d['label']:12s}  dist={dist_str}  "
              f"aspect={d['aspect']:.2f}  fill={d['fill']:.2f}")

    WIN = "Distance Estimation Result  [q]=quit [s]=save"
    cv.namedWindow(WIN, cv.WINDOW_NORMAL)
    cv.imshow(WIN, result)
    if cv.waitKey(0) == ord('s'):
        cv.imwrite("result.jpg", result)
        print("[INFO] Saved result.jpg")
    cv.destroyAllWindows()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODE = "camera"   # "camera" | "image"

    if MODE == "camera":
        run_camera()
    elif MODE == "image":
        run_image("test.jpg")

# Computer Vision

A progressive collection of **OpenCV + Python** computer vision experiments — from
fundamental image processing primitives to a fully calibrated, real-time **multi-object
distance estimation system** built for a Lenovo 300 USB webcam.

> **Author:** Shyam Hirpara  
> **Stack:** Python 3 · OpenCV 4 · NumPy · Matplotlib

---

## Repository Structure

```
Computer-Vision/
│
├── src/                            # Core learning experiments
│   ├── read_image.py               # Load and display a static image
│   ├── read_video.py               # Read and display a video file
│   ├── capture_video.py            # Live webcam capture
│   ├── drawings.py                 # Drawing shapes and text with OpenCV
│   ├── mouse_click_events.py       # Interactive mouse event handling
│   ├── arithmetic_operations.py    # Image blending, masking, bitwise ops
│   ├── basic_image_processing.py   # Resize, crop, flip, ROI operations
│   │
│   ├── image processing/           # Image processing techniques
│   │   ├── color_space_change.py       # BGR ↔ HSV ↔ Grayscale conversion (live)
│   │   ├── simple_thresholding.py      # 5 global threshold methods compared
│   │   ├── adaptive_thresholding.py    # Mean / Gaussian adaptive thresholding
│   │   ├── smoothing.py                # Blur, Gaussian, median, bilateral filters
│   │   ├── gradients.py                # Sobel, Laplacian, Canny edge detection
│   │   └── geomentric_transformation.py# Warp, rotate, translate, perspective
│   │
│   └── Detection/                  # Object detection experiments
│       ├── box_detector.py             # White box detection via HSV + contours
│       └── box_detector_explanation.md # Detailed algorithm walkthrough
│
├── lenovo300/                      # 📦 Full distance estimation system
│   ├── config.py                   # Central config (camera, objects, thresholds)
│   ├── object_detector.py          # LAB+Otsu detection & classification pipeline
│   ├── distance_estimator.py       # Triangle Similarity distance + HUD overlay
│   ├── camera_calibration.py       # Interactive checkerboard calibration tool
│   ├── calibration_tuner.py        # Real-time fx/fy fine-tuning with trackbars
│   ├── generate_checkerboard.py    # Generates printable calibration board
│   ├── checkerboard_9x6_A4.png     # Pre-generated 9×6 board (print on A4)
│   └── camera_calibration.npz      # Saved calibration matrix (K + distortion)
│
├── solve_pnp/                      # solvePnP research scratch space
├── requirements.txt
└── README.md
```

---

## Modules

### `src/` — Core Experiments

Standalone scripts covering fundamental OpenCV concepts, each self-documenting
with inline comments.

| Script | Concept |
|--------|---------|
| `read_image.py` | `cv.imread`, `cv.imshow`, window management |
| `read_video.py` | `cv.VideoCapture` with file input |
| `capture_video.py` | Live webcam feed |
| `drawings.py` | Lines, circles, rectangles, text overlay |
| `mouse_click_events.py` | `cv.setMouseCallback`, coordinate tracking |
| `arithmetic_operations.py` | `cv.addWeighted`, `cv.bitwise_and/or/xor` |
| `basic_image_processing.py` | Resize, crop, channel split, ROI extraction |

---

### `src/image processing/` — Processing Techniques

| Script | Techniques |
|--------|-----------|
| `color_space_change.py` | Live BGR→HSV→Gray, `cv.inRange` white detection |
| `simple_thresholding.py` | Binary, binary-inv, truncate, to-zero, Otsu |
| `adaptive_thresholding.py` | Mean vs Gaussian adaptive, block size effects |
| `smoothing.py` | Box blur, Gaussian, median, bilateral filter |
| `gradients.py` | Sobel X/Y, combined, Laplacian, Canny (live) |
| `geomentric_transformation.py` | Affine warp, rotation, translation, perspective transform |

---

### `src/Detection/` — Object Detection

#### `box_detector.py`
Detects **white rectangular boxes** on a contrasting background using:
- HSV colour thresholding
- Morphological cleanup (open/close)
- Contour detection + bounding box extraction
- Real-time webcam overlay

See [`box_detector_explanation.md`](src/Detection/box_detector_explanation.md) for a
detailed algorithm walkthrough.

---

### `lenovo300/` — Distance Estimation System

A production-quality module built for the **Lenovo 300 FHD USB Webcam (95° FOV)**.

#### What it does
Detects and classifies white 3D objects (Small Cube, Large Cube, Cylinder), estimates
their real-world distance from the camera, and overlays a HUD on the live feed.

#### Pipeline
```
Frame → Undistort → LAB L-channel → Otsu threshold + min-L guard
      → Morphological cleanup → Contour detection → Shape classification
      → Triangle Similarity distance → solvePnP pose → HUD overlay
```

#### Key features
- **Lighting-robust detection** — LAB+Otsu self-calibrates to ambient brightness every
  frame (no manual HSV tuning)
- **Accurate distance off-centre** — ray-angle Euclidean correction + tilt compensation
  for square objects
- **Full HUD** — corner-bracket bbox, pill-shaped distance badges, numbered object IDs,
  3D axes, pose (pitch/yaw/roll), FPS chip, dark status panel
- **Camera calibration tools** — interactive checkerboard session + real-time focal
  length tuner

#### Run
```bash
python lenovo300/distance_estimator.py
```

#### Keys
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save snapshot |
| `m` | Toggle binary mask view |
| `a` | Toggle 3D axes |
| `p` | Toggle pose text |
| `c` | Print detections to terminal |

#### Calibration
```bash
# Step 1 — run interactive calibration (print checkerboard_9x6_A4.png first)
python lenovo300/camera_calibration.py

# Step 2 — optional fine-tuning
python lenovo300/calibration_tuner.py
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/ShyamHirpara/Computer-Vision.git
cd Computer-Vision

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:**
```
opencv-python
numpy
matplotlib
```

---

## Usage Examples

```bash
# Basic image processing
python src/basic_image_processing.py

# Live edge detection
python src/image\ processing/gradients.py

# White box detector
python src/Detection/box_detector.py

# Full distance estimator (lenovo300 webcam)
python lenovo300/distance_estimator.py
```

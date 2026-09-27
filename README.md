# Smart Traffic Monitor

A real-time computer vision system that watches traffic video, tracks every
vehicle, estimates its actual speed in km/h, reads its license plate, and
automatically logs speeding violations with photo evidence — plus a
dashboard to review them.

This isn't a toy classifier. It's a full pipeline: **detection → tracking →
geometric speed estimation → OCR → persistent logging → visualization** —
the same category of system used in real traffic-enforcement cameras.

## How it works

1. **Vehicle detection & tracking** (`vehicle_detector.py`) — YOLOv8
   detects cars, motorcycles, buses, and trucks in each frame. Ultralytics'
   built-in ByteTrack assigns a stable ID to each vehicle so we know
   "vehicle #17 in frame 100" is the same as "vehicle #17 in frame 150."

2. **Speed estimation** (`utils/geometry.py`) — Pixel distance isn't real
   distance, because the camera is at an angle (perspective). We fix this
   with a **homography**: four points on the road that we know the real
   width/height of (e.g. a marked lane segment) are mapped to a flat,
   bird's-eye rectangle. Once every pixel can be converted to real-world
   meters, tracking a vehicle's position over time gives distance ÷ time =
   speed.

3. **License plate OCR** (`plate_reader.py`) — EasyOCR reads the plate
   from the lower part of each vehicle crop. This runs only every N frames
   per vehicle since OCR is the most expensive step.

4. **Violation logging** (`violation_logger.py`) — When a tracked
   vehicle's speed exceeds the configured limit, its plate, speed,
   timestamp, and a cropped snapshot are saved to a SQLite database —
   once per vehicle, not once per frame.

5. **Dashboard** (`dashboard.py`) — A Streamlit app to browse logged
   violations with snapshots, without touching the database directly.

## Setup

```bash
pip install -r requirements.txt
```

The first run of `main.py` will auto-download the small pretrained YOLOv8
model (`yolov8n.pt`).

## Calibration (the part that makes this "real")

Open one frame of your video and identify four points that form a
rectangle on the road surface whose real-world size you know (e.g. the
painted lines of a single lane, or the width between two lamp posts).
Update these in `config.py`:

```python
SOURCE_POINTS = [(550, 400), (1150, 400), (1750, 900), (-50, 900)]
REAL_WORLD_WIDTH_M = 10.0
REAL_WORLD_HEIGHT_M = 40.0
```

Bad calibration is the #1 cause of wrong speed readings — spend a few
minutes getting this right on your footage.

## Running it

```bash
python main.py
```

Point `VIDEO_SOURCE` in `config.py` at your own traffic footage, a webcam
(`0`), or an RTSP camera stream.

To review violations after a run:

```bash
streamlit run dashboard.py
```

## Where this goes from here (great interview talking points)

- Swap the generic vehicle-detector OCR crop for a **dedicated license-plate
  detection model** fine-tuned on a plate dataset — big accuracy jump.
- Add **red-light violation detection** using a defined stop-line polygon.
- Deploy on an edge device (Jetson Nano/Orin) with a quantized model for
  real deployments at intersections.
- Add **re-identification** so a vehicle that briefly leaves frame and
  re-enters isn't treated as a new ID.
- Wrap `main.py` as a FastAPI service so multiple camera feeds can push
  violations into one central database.

## Why this project stands out to employers

Most beginner CV portfolios stop at "classify an image." This project
demonstrates the full skill stack hiring managers actually look for:
object detection, multi-object tracking, geometric computer vision
(homography), OCR integration, a persistence layer, and a usable frontend
— all working together to solve a problem with obvious commercial value
(traffic enforcement, smart cities, parking systems, fleet monitoring).

"""
Configuration for Smart Traffic Monitor.
Edit these values to match your camera setup and desired thresholds.
"""

# --- Video source ---
VIDEO_SOURCE = "sample_traffic.mp4"   # path to a video file, or 0 for a webcam, or an RTSP url

# --- Detection ---
VEHICLE_MODEL_PATH = "yolov8n.pt"     # pretrained COCO model (detects car, truck, bus, motorcycle)
VEHICLE_CLASSES = [2, 3, 5, 7]        # COCO class ids: car, motorcycle, bus, truck
CONFIDENCE_THRESHOLD = 0.4

# --- Speed estimation ---
# Four points (in pixels, from the video frame) marking a rectangle of KNOWN real-world
# size on the road -- e.g. the corners of a marked lane segment.
# Order: top-left, top-right, bottom-right, bottom-left.
SOURCE_POINTS = [(550, 400), (1150, 400), (1750, 900), (-50, 900)]

# Real-world width/height of that rectangle in meters (measure once on-site, or estimate
# from lane width standards / satellite imagery).
REAL_WORLD_WIDTH_M = 10.0
REAL_WORLD_HEIGHT_M = 40.0

SPEED_LIMIT_KMH = 60

# --- Plate OCR ---
RUN_OCR_EVERY_N_FRAMES = 5    # OCR is expensive -- only run it periodically per tracked vehicle
OCR_MIN_CONFIDENCE = 0.4

# --- Violation logging ---
LOG_DB_PATH = "violations.db"
SNAPSHOT_DIR = "snapshots"

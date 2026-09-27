import time
import cv2

import config
from vehicle_detector import VehicleDetector
from geometry import SpeedEstimator
from plate_reader import PlateReader
from violation_logger import ViolationLogger


def foot_point(box):
    """Bottom-center of the box = where the vehicle meets the road plane."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, y2)


def main():
    detector = VehicleDetector()
    speed_estimator = SpeedEstimator(
        config.SOURCE_POINTS, config.REAL_WORLD_WIDTH_M, config.REAL_WORLD_HEIGHT_M
    )
    plate_reader = PlateReader()
    logger = ViolationLogger()

    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    frame_idx = 0
    plate_cache = {}   # track_id -> last confidently-read plate text

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = time.time()
        detections = detector.track(frame)

        for det in detections:
            track_id, box = det["id"], det["box"]
            speed_kmh = speed_estimator.update(track_id, foot_point(box), timestamp)

            plate_text = plate_cache.get(track_id)
            if frame_idx % config.RUN_OCR_EVERY_N_FRAMES == 0:
                x1, y1, x2, y2 = box
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                text, conf = plate_reader.read(crop)
                if text:
                    plate_cache[track_id] = text
                    plate_text = text

            logger.maybe_log_speeding(track_id, plate_text, speed_kmh, frame, box)

            # --- draw overlay ---
            x1, y1, x2, y2 = box
            is_speeding = speed_kmh is not None and speed_kmh > config.SPEED_LIMIT_KMH
            color = (0, 0, 255) if is_speeding else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID {track_id}"
            if speed_kmh is not None:
                label += f" {speed_kmh:.0f} km/h"
            if plate_text:
                label += f" [{plate_text}]"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Smart Traffic Monitor", frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

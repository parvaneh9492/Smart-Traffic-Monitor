from ultralytics import YOLO
import config


class VehicleDetector:
    """
    Wraps a YOLOv8 model to detect AND track vehicles across frames.
    Ultralytics' built-in .track() uses ByteTrack under the hood, which assigns
    a stable ID to each vehicle so we can follow it across multiple frames --
    essential for speed estimation and for not double-counting violations.
    """

    def __init__(self):
        self.model = YOLO(config.VEHICLE_MODEL_PATH)

    def track(self, frame):
        """
        Runs detection + tracking on a single frame.
        Returns a list of dicts: {id, box (x1, y1, x2, y2), class_name, conf}
        """
        results = self.model.track(
            frame,
            classes=config.VEHICLE_CLASSES,
            conf=config.CONFIDENCE_THRESHOLD,
            persist=True,      # keep the same track IDs alive across calls
            verbose=False,
        )[0]

        detections = []
        if results.boxes.id is None:
            return detections

        for box, track_id, cls, conf in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.id.cpu().numpy(),
            results.boxes.cls.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
        ):
            detections.append({
                "id": int(track_id),
                "box": tuple(map(int, box)),
                "class_name": self.model.names[int(cls)],
                "conf": float(conf),
            })
        return detections

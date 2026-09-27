import numpy as np
import cv2


class SpeedEstimator:
    """
    Converts pixel movement into real-world speed (km/h) using a homography
    that maps the road plane in the camera image to a bird's-eye rectangle
    of known real-world size. This is the same core technique used in real
    traffic-camera speed enforcement systems.
    """

    def __init__(self, source_points, real_width_m, real_height_m):
        dst_points = np.array([
            [0, 0],
            [real_width_m, 0],
            [real_width_m, real_height_m],
            [0, real_height_m],
        ], dtype=np.float32)
        src = np.array(source_points, dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst_points)
        self.track_history = {}   # track_id -> list of (timestamp, world_x, world_y)

    def _pixel_to_world(self, point_px):
        px = np.array([[point_px]], dtype=np.float32)
        world = cv2.perspectiveTransform(px, self.H)
        return world[0][0]   # (x_m, y_m)

    def update(self, track_id, foot_point_px, timestamp):
        """
        foot_point_px: (x, y) pixel location of the vehicle's contact point with
        the road (bottom-center of its bounding box works well in practice).
        Returns speed in km/h, or None if there isn't enough history yet.
        """
        world_xy = self._pixel_to_world(foot_point_px)
        history = self.track_history.setdefault(track_id, [])
        history.append((timestamp, world_xy[0], world_xy[1]))

        if len(history) > 30:        # keep roughly the last second at 30 fps
            history.pop(0)

        if len(history) < 5:
            return None

        t0, x0, y0 = history[0]
        t1, x1, y1 = history[-1]
        dt = t1 - t0
        if dt <= 0:
            return None

        dist_m = np.hypot(x1 - x0, y1 - y0)
        speed_mps = dist_m / dt
        return speed_mps * 3.6

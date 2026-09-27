import re
import easyocr
import config


class PlateReader:
    """
    Reads license plate text from a cropped vehicle image.

    This version runs EasyOCR directly on the lower portion of the vehicle
    crop, which works reasonably well for front/rear-facing traffic cameras.
    For a production deployment, swap in a dedicated plate-detector model
    (e.g. a YOLOv8 model fine-tuned on a license-plate dataset) to crop the
    exact plate region before OCR -- that will noticeably improve accuracy.
    """

    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=True)

    def read(self, vehicle_crop):
        if vehicle_crop.size == 0:
            return None, 0.0

        h = vehicle_crop.shape[0]
        lower_region = vehicle_crop[int(h * 0.55):, :]   # plates sit low on most vehicles

        results = self.reader.readtext(lower_region)
        best_text, best_conf = None, 0.0

        for _, text, conf in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            if len(cleaned) >= 4 and conf > best_conf:
                best_text, best_conf = cleaned, conf

        if best_conf < config.OCR_MIN_CONFIDENCE:
            return None, best_conf

        return best_text, best_conf

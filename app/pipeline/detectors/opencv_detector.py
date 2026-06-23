import os
import logging
import cv2
from .change_white_booths import recolor
from .booth_detector import detect as detect_booths_cv, extract_labels_with_ocr, Params

logger = logging.getLogger(__name__)

class OpenCVDetector:
    def __init__(self, run_ocr=True):
        """
        Initializes the OpenCV Structural Pass parameters with the STRICT rules
        discovered during hyper-tuning.
        """
        logger.debug("__init__() called with run_ocr=%s", run_ocr)
        self.run_ocr = run_ocr
        self.params = Params(
            min_area_frac=1e-4,        # Lowered to catch 40x40 booths like DS1
            max_area_frac=0.06,
            min_side_px=20,            # Require longer minimum sides
            min_rect_score=0.60,       # Strict rectangle requirement
            line_len_frac=0.03,        # Require longer cut lines
            cut_contrast=22,           # Slightly lower to catch faint internal dividers
            enable_subdivide=True      # Split large blocks along internal dividers
        )

    def detect(self, image_path):
        """
        Executes the full pipeline:
        1. Recolors white-on-white grid blocks
        2. Detects booth geometries mathematically
        3. Extracts text via OCR (if run_ocr is True)
        4. FILTERS OUT any booth that does not contain text (if run_ocr is True)
        """
        logger.debug("detect() called with image_path=%s", image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Could not load image at {image_path}")

        base_name = os.path.basename(image_path)
        out_dir = os.path.dirname(image_path) or "."
        temp_recolored = os.path.join(out_dir, f"temp_{base_name}")

        # 2. Geometric Detection
        logger.debug("detect: running geometric detection (detect_booths_cv)")
        booths, meta = detect_booths_cv(image_path, self.params)
        logger.info("detect: geometric detection found %d raw booths", len(booths))

        # 3. OCR Text Extraction
        if self.run_ocr:
            logger.debug("detect: running OCR label extraction on %d booths", len(booths))
            original_bgr = cv2.imread(image_path)
            extract_labels_with_ocr(original_bgr, booths)

        # Clean up temp file
        if os.path.exists(temp_recolored):
            logger.debug("detect: removing temp file %s", temp_recolored)
            os.remove(temp_recolored)

        final_boxes = []
        # 4. Strict Filtering
        for b in booths:
            label = getattr(b, 'name', "") if hasattr(b, 'name') else ""
            if self.run_ocr and (not label or str(label).strip() == ""):
                continue

            x, y, w, h = b.bbox
            if getattr(b, "coords", None):
                coords = [list(p) for p in b.coords]
                if coords[0] != coords[-1]:
                    coords = coords + [coords[0]]
            else:
                coords = [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]
            final_boxes.append({
                "coordinates": coords,
                "bbox": b.bbox,
                "label": label,
                "centroid": b.centroid,
                "score": 1.0,
                "source": "opencv_strict"
            })

        logger.info("detect: returning %d booths after strict filtering (run_ocr=%s)",
                    len(final_boxes), self.run_ocr)
        return final_boxes

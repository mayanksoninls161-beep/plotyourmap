import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, label="Booth"):
    """
    Draw bounding boxes on the image.
    boxes: list of dicts like {"id": 1, "bbox": [x1, y1, x2, y2]}
    Returns the modified image.
    """
    logger.debug("draw_boxes() called with %d boxes, label=%r", len(boxes) if boxes else 0, label)
    # Create a copy so we don't modify the original
    img_draw = image.copy()

    for b in boxes:
        bbox = b['bbox']
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        if b.get('id') is not None:
            text = f"{label} {b['id']}"
            cv2.putText(img_draw, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
    return img_draw

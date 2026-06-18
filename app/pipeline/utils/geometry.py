import numpy as np
import cv2


def _poly_area(poly):
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    return float(abs(cv2.contourArea(p)))


def polygon_iou(poly1, poly2):
    """IoU of two CONVEX polygons (e.g. oriented booth quads from minAreaRect).

    Tilted booths overlap far less by their true footprint than by their loose
    axis-aligned envelope, so comparing the quads here is what stops fusion/NMS
    from wrongly merging tilted neighbours. Returns 0.0 on degenerate input.
    """
    a = np.asarray(poly1, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(poly2, dtype=np.float32).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except Exception:
        return 0.0
    if inter <= 0:
        return 0.0
    ua = _poly_area(a) + _poly_area(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def polygon_overlap(poly1, poly2):
    """Return (iou, ios) for two CONVEX polygons in ONE intersection call.

    ios = intersection-over-smaller = inter / min(area1, area2). It stays high
    when one quad is mostly SWALLOWED by the other even though their IoU is low
    because the areas differ a lot -- e.g. a label box / icon / logo mark
    detected as a tiny cell sitting inside the real booth cell. IoU alone cannot
    suppress that (a 7x-smaller nested box has IoU ~0.15), so NMS needs ios too.
    """
    a = np.asarray(poly1, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(poly2, dtype=np.float32).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0, 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except Exception:
        return 0.0, 0.0
    if inter <= 0:
        return 0.0, 0.0
    a1, a2 = _poly_area(a), _poly_area(b)
    union = a1 + a2 - inter
    iou = float(inter / union) if union > 0 else 0.0
    m = min(a1, a2)
    ios = float(inter / m) if m > 0 else 0.0
    return iou, ios


def _bbox_overlap(box1, box2):
    """(iou, ios) for two axis-aligned boxes [x1,y1,x2,y2]. ios as in
    `polygon_overlap`. Used as the fallback when a box carries no polygon."""
    iou = calculate_iou(box1, box2)
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return iou, 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    m = min(a1, a2)
    ios = float(inter / m) if m > 0 else 0.0
    return iou, ios


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / float(union_area) if union_area > 0 else 0.0

def non_max_suppression(boxes, iou_threshold=0.3, containment_threshold=0.7):
    """
    Apply Non-Maximum Suppression (NMS) to overlapping boxes.
    boxes: list of dicts like {"bbox": [x1, y1, x2, y2], "score": float}

    A box is suppressed by an already-kept box when EITHER
      * their IoU exceeds `iou_threshold` (the classic overlap test), OR
      * one is >= `containment_threshold` swallowed by the other (intersection
        over the SMALLER area). This second test removes nested duplicates that
        IoU misses -- a small label/icon cell sitting inside the real booth, or
        the same booth found by two passes at very different sizes -- without
        touching genuinely adjacent cells, which only share a border (ios ~ 0).

    Ordering: higher score wins, then LARGER area wins. The area tiebreak makes
    containment deterministic in the right direction -- the booth is kept and the
    smaller fragment nested inside it is dropped, never the reverse.

    Backward-compatible: callers passing only {"bbox","score"} (no "poly") and
    the previous single-arg signature still behave as before.
    """
    if not boxes:
        return []

    def _area(b):
        x1, y1, x2, y2 = b['bbox']
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    # Pre-processing: Drop coarse merged blocks that are tiled by >= 2 finer boxes
    # ONLY if the finer boxes are not just skinny text-slices.
    valid_boxes = []
    for big in boxes:
        covered, n_inside = 0.0, 0
        is_false_split = False
        
        for small in boxes:
            if small is big or _area(small) >= _area(big):
                continue
            if 'poly' in small and 'poly' in big:
                _, ios = polygon_overlap(small['poly'], big['poly'])
            else:
                _, ios = _bbox_overlap(small['bbox'], big['bbox'])
            
            if ios > 0.80:
                # Check if small box is a skinny text slice
                w, h = small['bbox'][2] - small['bbox'][0], small['bbox'][3] - small['bbox'][1]
                aspect = max(w, h) / max(1.0, float(min(w, h)))
                if aspect > 2.5 and _area(small) < 15000:
                    is_false_split = True
                    
                covered += ios * _area(small)
                n_inside += 1
                
        if n_inside >= 2 and covered >= 0.75 * _area(big) and not is_false_split:
            print(f"Dropped by merged block: {_area(big)}"); continue # big is a true merged block, drop it!
        valid_boxes.append(big)

    # Sort by score desc, then area desc (so the larger of a nested pair is kept)
    sorted_boxes = sorted(valid_boxes, key=lambda x: (x.get('score', 1.0), _area(x)),
                          reverse=True)
    kept_boxes = []

    for current in sorted_boxes:
        should_keep = True
        evict_idx = None          # index into kept_boxes to replace
        for i, kept in enumerate(kept_boxes):
            if 'poly' in current and 'poly' in kept:
                iou, ios = polygon_overlap(current['poly'], kept['poly'])
            else:
                iou, ios = _bbox_overlap(current['bbox'], kept['bbox'])

            if iou > iou_threshold:
                # Classic overlap — higher-score (already-kept) wins.
                should_keep = False
                break

            if ios > containment_threshold:
                # Pure containment (one box swallowed by the other).
                # The LARGER box is the real booth; the smaller one is a
                # fragment / label / icon.  Always keep the bigger one.
                if _area(current) > _area(kept):
                    # Current is the big box — evict the small kept box,
                    # continue checking remaining kept boxes.
                    evict_idx = i
                else:
                    # Current is the small nested fragment — drop it.
                    should_keep = False
                    break

        if not should_keep:
            continue
        if evict_idx is not None:
            kept_boxes[evict_idx] = current
        else:
            kept_boxes.append(current)

    return kept_boxes

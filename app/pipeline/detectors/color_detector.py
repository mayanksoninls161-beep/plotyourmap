"""
Generic, color-agnostic booth-region detector.

Idea (works on ANY color-coded floor plan, not just pink):
  1. DISCOVER the dominant fill colors in the image (k-means over the colored,
     non-white, non-ink pixels). On AutoTech Asia this finds the pale-pink booth
     fill; on other plans it finds whatever color(s) the booths use.
  2. SEGMENT each discovered color into rectangular regions = candidate booths
     (this is the "find different colors and highlight them" step).
  3. OCR each region to LABEL it and CLASSIFY its type (booth vs a named zone
     like FOOD COURT / REGISTRATION / WC). OCR is a labeler/classifier here, not
     a recall gate -- a clean color-filled rectangle is kept as a booth even if
     its text is unreadable, which keeps recall high. (Set ocr_gate=True to drop
     regions with no readable text; off by default while OCR mangles the
     stylized dark-on-pink labels and would wrongly reject valid booths.)

Output matches the rest of the pipeline: a list of dicts with
  {coordinates, bbox(xywh), label, centroid, score, source, color, type}

This module is ADDITIVE -- it does not modify any existing detector.
"""
import os
import logging
import cv2
import numpy as np
import concurrent.futures

logger = logging.getLogger(__name__)

# Reuse the pipeline's own OCR (Tesseract) so labeling stays consistent.
try:
    from .booth_detector import _ocr_booth_name
    _OCR_AVAILABLE = True
except Exception:                                    # pragma: no cover
    _OCR_AVAILABLE = False
    def _ocr_booth_name(bgr, bbox):
        return None

# Words that mark a colored block as a NAMED ZONE rather than a sellable booth.
ZONE_KEYWORDS = {
    "food", "court", "registration", "register", "entry", "exit", "entrance",
    "wc", "toilet", "washroom", "restroom", "pavilion", "lounge", "cafe",
    "cafeteria", "stage", "office", "store", "storage", "stair", "stairs",
    "steps", "lift", "elevator", "escalator", "reception", "help", "info",
    "information", "parking", "prayer", "medical", "aid", "atm", "meeting",
    "conference", "seminar", "organiser", "organizer", "lobby", "passage",
    "walkway", "aisle", "hall", "gate",
}


def _is_zone(label: str) -> bool:
    logger.debug("_is_zone() called with label=%r", label)
    if not label:
        return False
    toks = [t.lower() for t in label.replace("/", " ").split()]
    return any(t in ZONE_KEYWORDS for t in toks)


class ColorDetector:
    def __init__(self,
                 max_colors: int = 8,
                 min_color_frac: float = 3e-3,   # a fill color must cover >=0.3% of the image
                 min_area_frac: float = 2e-4,    # smallest booth (frac of image area);
                                                 # 3e-4 cut ~72 real small booths at
                                                 # zero noise gain (min_side_px does the
                                                 # thin-noise filtering, not this floor)
                 max_area_frac: float = 0.06,    # biggest single region kept
                 min_side_px: int = None,        # None => dynamic: sqrt(min booth area),
                                                 # so the thin-strip cutoff scales with
                                                 # image resolution (==20px on a 1224x1584
                                                 # plan) instead of a fixed pixel count
                 min_fill: float = 0.45,         # region area / bbox area (rectangular-ish)
                 color_dist: float = 45.0,       # BGR L2 radius around a discovered color
                 run_ocr: bool = True,
                 ocr_gate: bool = False,         # if True, drop regions with no readable OCR text
                 neutral_gray: tuple = None,      # (lo, hi) gray band for desaturated "neutral"
                                                  # grey booths; None => skip them (default).
                 close_ksize: int = 9):           # MORPH_CLOSE kernel that fills dark lines (X
                                                  # marks) inside a fill; 9 bridges thin dividers
                                                  # too, FUSING abutting same-color cells. Shrink
                                                  # (e.g. 3) on dense grids so cells stay split.
        logger.debug("__init__() called with max_colors=%s, min_area_frac=%s, "
                     "max_area_frac=%s, run_ocr=%s, ocr_gate=%s, close_ksize=%s",
                     max_colors, min_area_frac, max_area_frac, run_ocr, ocr_gate, close_ksize)
        self.max_colors = max_colors
        self.min_color_frac = min_color_frac
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.min_side_px = min_side_px
        self.min_fill = min_fill
        self.color_dist = color_dist
        self.run_ocr = run_ocr
        self.ocr_gate = ocr_gate
        self.neutral_gray = neutral_gray
        self.close_ksize = max(1, int(close_ksize))

    # ---------------------------------------------------------------- colors
    def _discover_colors(self, bgr):
        """Return [(center_bgr(float32[3]), coverage_frac), ...] for dominant fills.

        Centers are learned from DEEP INTERIORS only (eroded colored mask) so the
        k-means lands on the true solid fill color instead of being pulled toward
        anti-aliased edges, thin colored rules, or text-adjacent pixels -- those
        impure samples were desaturating the centers and tanking recall.
        """
        logger.debug("_discover_colors() called with bgr shape=%s", bgr.shape)
        H, W = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        colored = (S >= 20) & (V >= 60) & (gray >= 70) & (gray <= 252)
        if self.neutral_gray is not None:
            logger.debug("_discover_colors: including neutral_gray band %s", self.neutral_gray)
            lo, hi = self.neutral_gray
            colored = colored | ((S < 32) & (gray >= lo) & (gray <= hi))
        near_white = (S < 18) & (V > 235)
        mask = (colored & (~near_white)).astype(np.uint8)
        logger.debug("_discover_colors: colored mask has %d pixels", int(mask.sum()))
        if int(mask.sum()) < 50:
            logger.debug("_discover_colors: too few colored pixels, returning no colors")
            return []

        # sample only solid interiors for the k-means (drops thin strokes/edges)
        interior = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
        if int(interior.sum()) < 50:
            logger.debug("_discover_colors: eroded interior too small, falling back to full mask")
            interior = mask
        pts = bgr[interior.astype(bool)].reshape(-1, 3).astype(np.float32)
        rng = np.random.RandomState(0)
        if len(pts) > 30000:
            logger.debug("_discover_colors: subsampling %d points to 30000 for k-means", len(pts))
            pts = pts[rng.choice(len(pts), 30000, replace=False)]
        K = int(min(self.max_colors, max(2, len(pts) // 500)))
        logger.debug("_discover_colors: running k-means with K=%d on %d points", K, len(pts))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, centers = cv2.kmeans(pts, K, None, crit, 3, cv2.KMEANS_PP_CENTERS)

        # coverage = share of the FULL colored mask within color_dist of each center
        # (mirrors exactly what _regions_for_color will later capture)
        full = bgr[mask.astype(bool)].reshape(-1, 3).astype(np.int32)
        cand = []
        for c in centers:
            within = np.linalg.norm(full - c.astype(np.int32), axis=1) < self.color_dist
            cand.append((c, float(within.sum()) / float(H * W)))
        logger.debug("_discover_colors: %d candidate centers before merge", len(cand))

        # merge near-duplicate color centers, keep the better-covered one
        merged = []
        for c, cov in sorted(cand, key=lambda z: -z[1]):
            if any(np.linalg.norm(c - mc) < 35 for mc, _ in merged):
                continue
            merged.append((c, cov))
        result = [(c, cov) for c, cov in merged if cov >= self.min_color_frac]
        logger.info("_discover_colors: %d merged centers -> %d dominant fill colors above min_color_frac",
                    len(merged), len(result))
        return result

    # ---------------------------------------------------------------- regions
    def _regions_for_color(self, bgr, dark_d, center):
        """Connected rectangular regions of one fill color."""
        logger.debug("_regions_for_color() called with center=%s", tuple(int(v) for v in center))
        H, W = bgr.shape[:2]
        img_area = H * W
        # smallest allowed short side. A region thinner than a square of the
        # minimum booth area is a strip/divider, not a booth. Derived from the
        # image size so the cutoff scales with resolution; override by passing an
        # explicit min_side_px.
        min_side = (self.min_side_px if self.min_side_px is not None
                    else max(6, int(round((self.min_area_frac * img_area) ** 0.5))))
        dist = np.linalg.norm(bgr.astype(np.int32) - center.astype(np.int32), axis=2)
        m = (dist < self.color_dist).astype(np.uint8) * 255
        
        # close the raw color mask to fill in thin dark lines (like the X) inside it.
        # NOTE: too large a kernel also bridges the dividers BETWEEN abutting same-
        # color cells, fusing a dense row into one blob -- close_ksize controls this.
        k = self.close_ksize
        m_closed = (m if k <= 1 else
                    cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)))
        
        core = cv2.morphologyEx(m_closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=4)
        logger.debug("_regions_for_color: %d connected components (incl. background)", n)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (img_area * self.min_area_frac <= area <= img_area * self.max_area_frac):
                continue

            # ORIENTED rectangle for this component. minAreaRect of an
            # axis-aligned blob returns ~0 deg with sides == its AABB, so
            # straight booths are unaffected; a tilted booth gets a tight,
            # correctly-angled box instead of a loose axis-aligned one.
            ys, xs = np.where(labels[y:y + h, x:x + w] == i)
            pts = np.column_stack((xs + x, ys + y)).astype(np.int32)
            rect = cv2.minAreaRect(pts)                  # ((cx,cy),(rw,rh),angle)
            (rcx, rcy), (rw, rh), ang = rect
            if rw < 1 or rh < 1:
                continue
            short, long_ = min(rw, rh), max(rw, rh)
            # filters use the TRUE oriented side lengths + fill (vs the loose
            # axis-aligned bbox) -- this is what lets a ~45 deg booth survive.
            if short < min_side:
                continue
            fill = float(area / (rw * rh))
            if fill < self.min_fill:
                continue
            if short / float(long_) < 0.12:
                continue

            # fold angle to [0,45]; snap near-axis blobs back to a clean AABB so
            # the ~99% of straight booths stay pixel-identical (no quad jitter).
            a45 = abs(ang) % 90.0
            a45 = min(a45, 90.0 - a45)
            if a45 <= 7.0:
                quad = [[int(x), int(y)], [int(x + w), int(y)],
                        [int(x + w), int(y + h)], [int(x), int(y + h)]]
                bx, by, bw, bh = int(x), int(y), int(w), int(h)
            else:
                box = cv2.boxPoints(rect)
                quad = [[int(round(px)), int(round(py))] for px, py in box]
                qx = [p[0] for p in quad]
                qy = [p[1] for p in quad]
                bx, by = min(qx), min(qy)
                bw, bh = max(qx) - bx, max(qy) - by

            out.append({
                "quad": quad,                                    # oriented corners
                "bbox_xywh": (bx, by, bw, bh),                   # axis-aligned envelope
                "area": float(area),
                "centroid": (float(rcx), float(rcy)),
                "fill": fill,
                "angle": round(float(a45), 1),
                "color": tuple(int(v) for v in center[::-1]),    # store as RGB
            })
        logger.debug("_regions_for_color: kept %d regions for this color", len(out))
        return out

    @staticmethod
    def _dedupe(cands, iou_t=0.45):
        """Keep the largest region among overlapping ones (across color bands).

        Uses oriented-quad IoU so two tilted neighbours are not wrongly merged
        by their loose axis-aligned envelopes.
        """
        logger.debug("_dedupe() called with %d candidates, iou_t=%s", len(cands), iou_t)
        from utils.geometry import polygon_iou
        kept = []
        for c in sorted(cands, key=lambda z: -z["area"]):
            if all(polygon_iou(c["quad"], k["quad"]) <= iou_t for k in kept):
                kept.append(c)
        logger.info("_dedupe: %d candidates -> %d after IoU dedup", len(cands), len(kept))
        return kept

    # ---------------------------------------------------------------- public
    def detect(self, image_path):
        logger.debug("detect() called with image_path=%s", image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = (gray < 110).astype(np.uint8) * 255
        dark_d = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1)

        # Segment on the RAW image. A medianBlur here fills the 1px separators
        # between abutting same-color cells, fusing them into blobs (173 -> 64
        # booths on AutoTech Asia). Erosion inside _discover_colors already keeps
        # the learned color centers clean, so no pre-smoothing is needed.
        colors = self._discover_colors(bgr)
        logger.debug("detect: segmenting regions for %d discovered colors", len(colors))
        cands = []
        for center, _cov in colors:
            cands += self._regions_for_color(bgr, dark_d, center)
        logger.debug("detect: %d candidate regions before dedup", len(cands))
        cands = self._dedupe(cands)

        # OCR labeling + booth/zone classification
        if self.run_ocr and _OCR_AVAILABLE and cands:
            logger.debug("detect: running OCR labeling on %d candidates", len(cands))
            def _label(c):
                logger.debug("_label() called for bbox=%s", c["bbox_xywh"])
                x, y, w, h = c["bbox_xywh"]
                try:
                    c["label"] = _ocr_booth_name(bgr, (x, y, w, h)) or ""
                except Exception:
                    logger.exception("_label: OCR failed for bbox=%s", c.get("bbox_xywh"))
                    c["label"] = ""
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(_label, cands))
            # Hard OCR gate (same rule as the geometric pass): a region with no
            # readable text is not a sellable booth -- blank stalls, structural
            # pillars, banners -- so drop it. Only applied when OCR actually ran,
            # otherwise we'd discard everything.
            if self.ocr_gate:
                before = len(cands)
                cands = [c for c in cands if c["label"].strip()]
                logger.debug("detect: ocr_gate dropped %d candidates with no text (%d -> %d)",
                             before - len(cands), before, len(cands))
        else:
            logger.debug("detect: OCR skipped (run_ocr=%s, available=%s, cands=%d)",
                         self.run_ocr, _OCR_AVAILABLE, len(cands))
            for c in cands:
                c["label"] = ""

        out = []
        for c in cands:
            x, y, w, h = c["bbox_xywh"]
            coords = [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x, y]]  # Force perfectly straight rectangles!
            label = c.get("label", "")
            out.append({
                "coordinates": coords,
                "bbox": (x, y, w, h),                            # axis-aligned envelope
                "label": label,
                "centroid": c["centroid"],
                "score": round(0.5 + 0.5 * c["fill"], 3),
                "source": "color",
                "color": c["color"],
                "type": "zone" if _is_zone(label) else "booth",
                "angle": c.get("angle", 0.0),
            })
        logger.info("detect: returning %d color regions", len(out))
        return out

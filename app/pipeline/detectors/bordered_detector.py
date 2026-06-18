"""
Enclosed-cell ("bordered") booth detector -- recovers WHITE-ON-WHITE booths.

Why this pass exists
--------------------
The geometric pass (OpenCVDetector / booth_detector) finds booths by
ELIMINATION: it floods the page background inward, stopping only at DARK
borders, and keeps whatever is left. Faint reference-grid lines (V ~190-220) are
intentionally NOT flood barriers, so a block of white-on-white cells separated by
thin/faint borders floods through as one big background region and every cell
inside it is lost. The color pass ignores them too (its saturation gate drops
anything near-white). Result: dense white numbered grids (e.g. a hall full of
"110L .. 167L" available stalls) are invisible to BOTH existing passes.

This pass finds those cells the other way round -- by CONSTRUCTION, not
elimination. It traces booth BORDERS directly (Canny + adaptive threshold, both
of which fire on faint grey grid lines), seals them into a closed grid, and
treats each enclosed region as a candidate cell. A readable OCR label is
attached when present but is NOT a recall gate -- white "available" cells are
frequently blank -- so they survive.

Ported from the prior project's `floorplan_segmentation.py` (bt_4 "Mode B /
darken the border" + Canny/adaptive cell contours), with two upgrades carried
over from the rest of THIS package so the output is consistent with every other
detector:
  * ORIENTED cells (cv2.minAreaRect) -> tilted / diagonal grids get tight quads
    instead of loose axis-aligned envelopes;
  * an oriented-quad FILL test, so a ~45 deg cell is not rejected for the low
    axis-aligned "extent" a diamond necessarily has.

Contract (identical to every other detector in the package):
    detect(image_path) -> list[dict] with
    {coordinates, bbox(xywh), label, centroid, score, source, angle, type}

This module is ADDITIVE -- it imports from, but does not modify, any existing
detector.
"""
import os
import cv2
import numpy as np
import concurrent.futures

# Reuse the pipeline's own OCR (Tesseract) + zone classifier so labeling and
# booth/zone typing stay identical to the color pass.
try:
    from .booth_detector import _ocr_booth_name
    _OCR_AVAILABLE = True
except Exception:                                    # pragma: no cover
    _OCR_AVAILABLE = False
    def _ocr_booth_name(bgr, bbox):
        return None

try:
    from .color_detector import _is_zone
except Exception:                                    # pragma: no cover
    def _is_zone(label):
        return False


class BorderedCellDetector:
    def __init__(self,
                 min_area_frac: float = 1.5e-4,   # smallest cell (frac of image area)
                 max_area_frac: float = 0.06,     # biggest single cell (whole halls rejected)
                 min_side_px: int = None,         # None => dynamic sqrt(min cell area)
                 min_fill: float = 0.55,          # contourArea / minAreaRect area (oriented)
                 min_aspect: float = 0.14,        # short / long oriented side
                 max_corners: int = 12,           # polygon-vertex cap (reject organic blobs)
                 canny_lo: int = 30,
                 canny_hi: int = 120,
                 adaptive_block: int = 31,        # Canny/adaptive contour block (Method B)
                 adaptive_c: int = 8,
                 wall_block_div: int = 60,        # H+V wall pass block = min(H,W)/this (Method C)
                 wall_c: int = 7,
                 wall_len_frac: float = 0.005,    # min H/V wall length, frac of min(H,W)
                 score: float = 0.45,             # < color (0.5..1.0) & geometric (1.0): loses ties
                 run_ocr: bool = True,
                 ocr_gate: bool = False,          # if True, drop cells with no readable text
                 require_white: bool = True,      # only fire when the image HAS white cells
                 white_sat_max: int = 30,         # HSV S below this AND
                 white_val_min: int = 200,        # HSV V above this  => a white-ish pixel
                 white_min_frac: float = 1e-3):   # need one clean white blob >= this frac
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.min_side_px = min_side_px
        self.min_fill = min_fill
        self.min_aspect = min_aspect
        self.max_corners = max_corners
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi
        self.adaptive_block = adaptive_block | 1     # must be odd
        self.adaptive_c = adaptive_c
        self.wall_block_div = wall_block_div
        self.wall_c = wall_c
        self.wall_len_frac = wall_len_frac
        self.score = score
        self.run_ocr = run_ocr
        self.ocr_gate = ocr_gate
        self.require_white = require_white
        self.white_sat_max = white_sat_max
        self.white_val_min = white_val_min
        self.white_min_frac = white_min_frac

    # ---------------------------------------------------------------- gate
    def _has_white_cells(self, bgr):
        """True iff the image contains at least one clean, booth-sized WHITE blob.

        This is the activation gate (ported from the prior project's
        `recolor_white_booths` sanity check). A fully color-coded plan with no
        white-on-white booths -- e.g. AutoTech Asia -- fails this test, so the
        pass contributes NOTHING and the well-tuned colored-map behaviour is left
        exactly as it was. The white PAGE background is excluded automatically: it
        is one huge component far above the booth-size cap."""
        H, W = bgr.shape[:2]
        img_area = float(H * W)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 1] < self.white_sat_max) &
                 (hsv[..., 2] > self.white_val_min)).astype(np.uint8) * 255
        n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > img_area * 0.05 or min(w, h) < 15:    # page bg / specks
                continue
            if area / float(w * h) < 0.85:                  # not a clean rectangle
                continue
            if max(w, h) / max(1, min(w, h)) > 6:           # too elongated (rule/strip)
                continue
            if area >= img_area * self.white_min_frac:
                return True
        return False

    # ---------------------------------------------------------------- masks
    def _bordered_interior(self, bgr):
        """bt_4 Mode B: extract H+V wall lines (ink-dark OR faint adaptive grey),
        seal them into a closed grid, return the INVERSE so each enclosed cell
        interior is its own white blob."""
        H, W = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = cv2.inRange(bgr, (0, 0, 0), (50, 50, 50))
        dark = cv2.dilate(dark, np.ones((3, 3), np.uint8), iterations=1)
        block = max(3, int(min(H, W) / self.wall_block_div)) | 1
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, block, self.wall_c)
        edges = cv2.bitwise_or(dark, adapt)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 1)
        L = max(8, int(self.wall_len_frac * min(H, W)))
        horiz = cv2.morphologyEx(edges, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
        vert = cv2.morphologyEx(edges, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
        grid = cv2.bitwise_or(horiz, vert)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 1)
        return cv2.bitwise_not(grid)

    def _candidate_contours(self, bgr):
        """Every booth-border contour from three complementary border tracers.
        Canny (Method A) is orientation-agnostic, so it is what captures a tilted
        / diagonal white grid; adaptive contours (B) and H+V walls (C) add the
        faint-grey-bordered cells the elimination pass floods over."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        contours = []

        # Method A: Canny edges -> close -> contours (bordered cells, any angle)
        edges = cv2.Canny(gray, self.canny_lo, self.canny_hi)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        cA, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours += list(cA)

        # Method B: adaptive threshold -> contours (filled / faint-border cells)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, self.adaptive_block, self.adaptive_c)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        cB, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours += list(cB)

        # Method C: H+V wall interior blobs (bt_4 Mode B)
        cC, _ = cv2.findContours(self._bordered_interior(bgr),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours += list(cC)
        return contours

    # ---------------------------------------------------------------- cells
    def _oriented_cell(self, contour, img_area, min_side):
        """Validate a contour as a booth cell and return an ORIENTED record, or
        None. Same OBB treatment as ColorDetector: minAreaRect for the tight quad
        + oriented fill, snap near-axis cells back to a clean AABB."""
        area = float(cv2.contourArea(contour))
        if not (img_area * self.min_area_frac <= area <= img_area * self.max_area_frac):
            return None
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            return None
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if not (3 <= len(approx) <= self.max_corners):
            return None

        rect = cv2.minAreaRect(contour)                  # ((cx,cy),(rw,rh),angle)
        (rcx, rcy), (rw, rh), ang = rect
        if rw < 1 or rh < 1:
            return None
        short, long_ = min(rw, rh), max(rw, rh)
        if short < min_side:
            return None
        fill = float(area / (rw * rh))                   # oriented fill (angle-invariant)
        if fill < self.min_fill:
            return None
        if short / float(long_) < self.min_aspect:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        a45 = abs(ang) % 90.0
        a45 = min(a45, 90.0 - a45)
        if a45 <= 7.0:                                   # straight: clean AABB, no jitter
            quad = [[int(x), int(y)], [int(x + w), int(y)],
                    [int(x + w), int(y + h)], [int(x), int(y + h)]]
            bx, by, bw, bh = int(x), int(y), int(w), int(h)
        else:                                            # tilted: tight oriented quad
            box = cv2.boxPoints(rect)
            quad = [[int(round(px)), int(round(py))] for px, py in box]
            qx = [p[0] for p in quad]
            qy = [p[1] for p in quad]
            bx, by = min(qx), min(qy)
            bw, bh = max(qx) - bx, max(qy) - by
        return {
            "quad": quad,
            "bbox_xywh": (bx, by, bw, bh),
            "area": area,
            "centroid": (float(rcx), float(rcy)),
            "fill": fill,
            "source": "bordered",
            "score": 1.5,
            "angle": round(float(a45), 1),
        }

    @staticmethod
    def _dedupe(cands, iou_t=0.45, cover_t=0.55, contain_t=0.85):
        """Granularity-aware de-duplication of oriented cells, in three stages.

        The three border tracers produce TWO different kinds of duplicate, which
        need OPPOSITE preferences, so a single "keep largest + containment" rule
        cannot do both (it collapses real grids; a single "keep smallest" rule
        keeps stray fragments). They are separated here:

        Stage 1 -- concentric ("twice in the border"). Canny (A) and adaptive (B)
            trace the SAME cell as two near-identical quads -> high IoU. Greedy
            IoU suppression, SMALLEST-first, keeps one. Genuinely adjacent cells
            only share an edge (IoU ~ 0) so they are untouched; oriented quads are
            what make that true for tilted neighbours.

        Stage 2 -- row-envelope (prefer FINE). Canny also traces the outer
            boundary of a whole ROW of cells as ONE long quad when the internal
            walls are faint. That envelope has LOW IoU with each cell it spans (so
            Stage 1 keeps it) but is TILED by them. Drop any survivor that is
            >= cover_t covered by the union of >= 2 strictly-smaller survivors.
            This keeps the fine cells and removes the coarse box -- the opposite
            of keep-largest, which would eat the grid.

        Stage 3 -- nested fragment (prefer COARSE). A lone inner box (label/icon
            contour) sitting inside ONE real cell has low IoU but ios ~ 1. With
            the envelopes already gone (Stage 2), any survivor that is >= contain_t
            swallowed by a LARGER survivor must be such a fragment -- real booths
            do not nest -- so drop it. Adjacent cells never trigger this (ios ~ 0).
        """
        if not cands:
            return []
        from utils.geometry import polygon_overlap

        # Stage 1: IoU dedup, smallest first.
        kept = []
        for c in sorted(cands, key=lambda z: z["area"]):
            if all(polygon_overlap(c["quad"], k["quad"])[0] <= iou_t for k in kept):
                kept.append(c)

        # Stage 2: drop a coarse box tiled by >= 2 finer survivors.
        survivors = []
        for big in kept:
            covered, n_inside = 0.0, 0
            for small in kept:
                if small is big or small["area"] >= big["area"]:
                    continue
                _, ios = polygon_overlap(small["quad"], big["quad"])
                if ios > 0.80:                       # this finer cell sits inside `big`
                    covered += ios * small["area"]   # ios = inter / small.area here
                    n_inside += 1
            if n_inside >= 2 and covered >= cover_t * big["area"]:
                continue                              # `big` is just the row envelope
            survivors.append(big)

        # Stage 3: drop a fine box almost wholly inside a single LARGER survivor.
        final = []
        for c in survivors:
            nested = any(c["area"] < k["area"] and
                         polygon_overlap(c["quad"], k["quad"])[1] > contain_t
                         for k in survivors if k is not c)
            if not nested:
                final.append(c)
        return final

    # ---------------------------------------------------------------- public
    def detect(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)

        # Activation gate: skip entirely on plans with no white-on-white booths so
        # this recall pass never perturbs a clean color-coded map.
        if self.require_white and not self._has_white_cells(bgr):
            return []

        H, W = bgr.shape[:2]
        img_area = float(H * W)
        min_side = (self.min_side_px if self.min_side_px is not None
                    else max(6, int(round((self.min_area_frac * img_area) ** 0.5))))

        cands = []
        for c in self._candidate_contours(bgr):
            cell = self._oriented_cell(c, img_area, min_side)
            if cell is not None:
                cands.append(cell)
        cands = self._dedupe(cands)

        # OCR labeling (NOT a recall gate unless ocr_gate=True). Done after dedupe
        # so Tesseract only runs on survivors.
        if self.run_ocr and _OCR_AVAILABLE and cands:
            def _label(c):
                x, y, w, h = c["bbox_xywh"]
                try:
                    c["label"] = _ocr_booth_name(bgr, (x, y, w, h)) or ""
                except Exception:
                    c["label"] = ""
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(_label, cands))
            if self.ocr_gate:
                cands = [c for c in cands if c["label"].strip()]
        else:
            for c in cands:
                c["label"] = ""

        out = []
        for c in cands:
            x, y, w, h = c["bbox_xywh"]
            quad = c["quad"]
            coords = [list(p) for p in quad] + [list(quad[0])]   # closed oriented polygon
            label = c.get("label", "")
            out.append({
                "coordinates": coords,
                "bbox": (x, y, w, h),                            # axis-aligned envelope
                "label": label,
                "centroid": c["centroid"],
                "score": round(self.score, 3),
                "source": "bordered",
                "type": "zone" if _is_zone(label) else "booth",
                "angle": c.get("angle", 0.0),
            })
        return out

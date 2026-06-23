"""
Production ensemble booth detector.

Fuses three independent passes and resolves their overlaps:

  * GEOMETRIC pass  -- the strict, OCR-gated OpenCVDetector. High precision: it
    only keeps clean rectangles that also carry readable text.
  * COLOR pass      -- the generic, color-agnostic ColorDetector. High recall:
    it segments every dominant fill color into booth-shaped regions, keeping a
    region even when its label is unreadable.
  * BORDERED pass   -- the BorderedCellDetector. Recovers WHITE-ON-WHITE booths
    that both passes above miss: the geometric flood swallows faint-bordered
    white cells as background, and the color pass ignores them (saturation
    gate). It traces booth borders by construction (Canny + adaptive threshold)
    so dense white numbered grids survive; not OCR-gated, so blank "available"
    cells are kept.

The candidate sets are unioned and passed through the pipeline's own NMS
(`utils.geometry.non_max_suppression`). Scores establish a strict precedence on
overlap: geometric (flat 1.0) beats color (0.5..1.0 by rectangular fill), which
beats bordered (0.45). So wherever passes agree on a booth the most precise box
survives; the bordered pass only ever WINS where it is the sole detector that
fired -- i.e. on the white cells nothing else can see.

This class exposes the SAME contract as every other detector in the package --
`detect(image_path) -> list[dict]` -- so it is a drop-in replacement anywhere a
single detector is used (e.g. the FastAPI booth endpoint in main.py).

Each returned booth is a superset of the single-detector schema:

    {
      "id":          <int>,                       # 1-based, fusion order
      "coordinates": [[x,y],...,[x,y]],           # closed polygon
      "bbox":        (x, y, w, h),
      "label":       "<ocr text or ''>",
      "centroid":    (cx, cy),
      "score":       <float>,
      "source":      "opencv_strict" | "color",
      "color":       (r, g, b),                    # color-sourced regions only
      "type":        "booth" | "zone",             # color-sourced regions only
    }

The extra keys are purely additive, so downstream consumers that only read
`centroid` / `bbox` (e.g. _build_hall_booth_map) behave exactly as before.
"""
import logging
from typing import Dict, List, Optional

from .opencv_detector import OpenCVDetector
from .color_detector import ColorDetector
from .bordered_detector import BorderedCellDetector
from utils.geometry import non_max_suppression

logger = logging.getLogger(__name__)


class EnsembleDetector:
    def __init__(self,
                 use_geometric: bool = True,
                 use_color: bool = True,
                 use_bordered: bool = True,
                 bordered_min_area_frac: float = 0.0,
                 max_box_area_frac: float = 0.0,
                 demerge_with_bordered: bool = False,
                 color_neutral_gray: tuple = None,
                 iou_threshold: float = 0.4,
                 geometric: Optional[OpenCVDetector] = None,
                 color: Optional[ColorDetector] = None,
                 bordered: Optional[BorderedCellDetector] = None):
        """
        use_geometric / use_color / use_bordered : toggle any pass off to run a
            subset through the same fusion/NMS path.
        bordered_min_area_frac : when > 0, drop any surviving BORDERED-source box
            whose area is below this fraction of the whole image. Bordered boxes
            only survive NMS where no other pass fired, and the small survivors
            are empty "available" white cells (net false positives); the large
            survivors are real un-OCR'able areas (activity zones, pavilions,
            food courts). This keeps the big ones and discards the small junk.
            0.0 (default) disables the filter -> unchanged behaviour.
        max_box_area_frac : when > 0, drop any GEOMETRIC- or COLOR-source box
            whose bounding box exceeds this fraction of the whole image, BEFORE
            fusion. A hall-outline that the geometric pass traces as one giant
            "booth" (score 1.0, largest area) would otherwise be kept first by
            NMS and then SWALLOW every real booth nested inside it via the
            containment rule (intersection-over-smaller > threshold) -- collapsing
            a whole dense hall to a single box. Pruning the oversized box up front
            lets the nested booths survive. BORDERED boxes are exempt so the
            intended big "zone" survivors above are preserved. 0.0 (default)
            disables the filter -> unchanged behaviour.
        demerge_with_bordered : when True, a GEOMETRIC/COLOR box that encloses
            >=2 BORDERED tiles (each clearly smaller than it and large enough to
            survive the bordered min-area filter) is treated as a merge of
            adjacent same-fill booths -- the color mask's closing bridges the
            thin border between abutting cells, fusing them into one region,
            while the bordered pass traces each cell and tiles it correctly. NMS
            alone keeps the single bigger box and deletes the tiles (containment
            rule). Dropping the merged box up front lets its tiles win NMS and
            become individual booths. Pairs with a low bordered_min_area_frac so
            the freed tiles are not then removed by the post-fusion filter.
            False (default) -> unchanged behaviour.
        color_neutral_gray : (lo, hi) gray band forwarded to the color pass so it
            also captures desaturated "neutral" grey booths (which the saturation
            gate otherwise skips). None (default) -> grey booths ignored, unchanged
            behaviour.
        iou_threshold : overlap above which the lower-scoring box is dropped.
        geometric / color / bordered : inject pre-configured detector instances;
            when None the package defaults (already hyper-tuned) are used.
        """
        logger.debug("__init__() called with use_geometric=%s, use_color=%s, use_bordered=%s, "
                     "iou_threshold=%s, bordered_min_area_frac=%s, max_box_area_frac=%s, "
                     "demerge_with_bordered=%s",
                     use_geometric, use_color, use_bordered, iou_threshold,
                     bordered_min_area_frac, max_box_area_frac, demerge_with_bordered)
        if not (use_geometric or use_color or use_bordered):
            raise ValueError("EnsembleDetector needs at least one pass enabled "
                             "(use_geometric, use_color and/or use_bordered).")
        self.use_geometric = use_geometric
        self.use_color = use_color
        self.use_bordered = use_bordered
        self.bordered_min_area_frac = bordered_min_area_frac
        self.max_box_area_frac = max_box_area_frac
        self.demerge_with_bordered = demerge_with_bordered
        self.iou_threshold = iou_threshold
        self.geometric = geometric or OpenCVDetector()
        self.color = color or ColorDetector(neutral_gray=color_neutral_gray)
        self.bordered = bordered or BorderedCellDetector()

    @staticmethod
    def _xyxy(b: Dict) -> List[float]:
        logger.debug("_xyxy() called with bbox=%s", b.get("bbox"))
        x, y, w, h = b["bbox"]
        return [x, y, x + w, y + h]

    @staticmethod
    def _poly(b: Dict):
        """Oriented quad (open, >=3 pts) from a booth's `coordinates`, for
        rotated-IoU fusion. Returns None when no usable polygon is present."""
        logger.debug("_poly() called with source=%s", b.get("source"))
        c = b.get("coordinates")
        if not c:
            return None
        pts = list(c)
        if len(pts) >= 2 and list(pts[0]) == list(pts[-1]):
            pts = pts[:-1]                       # drop closing duplicate
        if len(pts) < 3:
            return None
        return [[float(p[0]), float(p[1])] for p in pts]

    def _run_pass(self, name: str, detector, image_path: str) -> List[Dict]:
        """Run one sub-detector. A failure in one pass must not sink the other,
        so we log and degrade to an empty result instead of propagating."""
        logger.debug("_run_pass() called for name=%s, image_path=%s", name, image_path)
        try:
            out = detector.detect(image_path)
            logger.info("ensemble: %s pass -> %d regions", name, len(out))
            return out
        except Exception:
            logger.exception("ensemble: %s pass failed; continuing without it", name)
            return []

    def detect(self, image_path: str) -> List[Dict]:
        logger.debug("detect() called with image_path=%s (geometric=%s, color=%s, bordered=%s)",
                     image_path, self.use_geometric, self.use_color, self.use_bordered)
        geo = self._run_pass("geometric", self.geometric, image_path) if self.use_geometric else []
        col = self._run_pass("color", self.color, image_path) if self.use_color else []
        bor = self._run_pass("bordered", self.bordered, image_path) if self.use_bordered else []

        pool = list(geo) + list(col) + list(bor)
        logger.debug("detect: pooled candidates geo=%d + color=%d + bordered=%d -> pool=%d",
                     len(geo), len(col), len(bor), len(pool))

        # Image area, read once, shared by every size-based fusion guard below.
        img_area = None
        if (self.max_box_area_frac > 0.0 or self.bordered_min_area_frac > 0.0
                or self.demerge_with_bordered):
            import cv2
            _img = cv2.imread(image_path)
            if _img is not None:
                img_area = float(_img.shape[0] * _img.shape[1])
                logger.debug("detect: img_area=%.0f read for size-based fusion guards", img_area)

        # Prune hall-sized GEOMETRIC/COLOR boxes BEFORE fusion. Such a box (e.g.
        # a hall outline traced as one "booth") would be kept first by NMS
        # (score 1.0, largest area) and then suppress every real booth nested
        # inside it via the containment rule -- collapsing a dense hall to one
        # box. Bordered boxes are exempt (their big survivors are intended zones).
        dropped_oversized = 0
        if self.max_box_area_frac > 0.0 and img_area is not None:
            cap = self.max_box_area_frac * img_area
            pruned = []
            for b in pool:
                if not str(b.get("source", "")).startswith("bordered"):
                    x, y, w, h = b["bbox"]
                    if float(w) * float(h) > cap:
                        dropped_oversized += 1
                        continue
                pruned.append(b)
            pool = pruned
            logger.debug("detect: pre-fusion prune dropped %d oversized geo/color boxes -> pool=%d",
                         dropped_oversized, len(pool))

        # De-merge GEOMETRIC/COLOR boxes that the BORDERED pass has already split.
        # The color mask's closing fuses abutting same-fill booths into one
        # region; the bordered pass traces each cell and tiles that region
        # correctly. NMS alone keeps the single bigger box and deletes the tiles
        # (containment). Here, if a non-bordered box encloses >=2 bordered tiles
        # -- each clearly smaller than it and large enough to survive the bordered
        # min-area filter -- drop the merged box so its tiles win NMS and become
        # individual booths.
        demerged = 0
        if self.demerge_with_bordered and img_area is not None:
            min_tile = (self.bordered_min_area_frac * img_area
                        if self.bordered_min_area_frac > 0.0 else 0.0)
            bordered_boxes = [b for b in pool
                              if str(b.get("source", "")).startswith("bordered")]
            kept_pool = []
            for b in pool:
                if str(b.get("source", "")).startswith("bordered"):
                    kept_pool.append(b)
                    continue
                bx, by, bw, bh = b["bbox"]
                barea = float(bw * bh)
                tiles = 0
                for t in bordered_boxes:
                    tx, ty, tw, th = t["bbox"]
                    tarea = float(tw * th)
                    if tarea < min_tile:
                        continue                  # tile would be filtered anyway
                    if not (0.10 * barea <= tarea <= 0.70 * barea):
                        continue                  # not a sub-tile of this box
                    ix0, iy0 = max(tx, bx), max(ty, by)
                    ix1, iy1 = min(tx + tw, bx + bw), min(ty + th, by + bh)
                    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
                    if tarea > 0 and (iw * ih) / tarea > 0.70:
                        tiles += 1
                if tiles >= 2:
                    demerged += 1
                    continue                      # drop the merged box
                kept_pool.append(b)
            pool = kept_pool
            logger.debug("detect: de-merge dropped %d merged geo/color boxes split by bordered tiles -> pool=%d",
                         demerged, len(pool))

        nms_in = []
        for b in pool:
            entry = {"bbox": self._xyxy(b),
                     "score": float(b.get("score", 1.0)),
                     "_ref": b}
            poly = self._poly(b)
            if poly is not None:
                entry["poly"] = poly          # rotated-IoU when available
            nms_in.append(entry)


        logger.debug("detect: running NMS on %d boxes (iou_threshold=%s)",
                     len(nms_in), self.iou_threshold)
        kept = non_max_suppression(nms_in, iou_threshold=self.iou_threshold)
        fused = [k["_ref"] for k in kept]
        logger.debug("detect: NMS kept %d of %d boxes", len(fused), len(nms_in))



        dropped_small_bordered = 0
        if self.bordered_min_area_frac > 0.0 and img_area is not None:
            min_area = self.bordered_min_area_frac * img_area
            filtered = []
            for b in fused:
                if str(b.get("source", "")).startswith("bordered"):
                    x, y, w, h = b["bbox"]
                    if w * h < min_area:
                        dropped_small_bordered += 1
                        continue
                filtered.append(b)
            fused = filtered
            logger.debug("detect: post-fusion filter dropped %d small bordered boxes -> %d kept",
                         dropped_small_bordered, len(fused))

        for i, b in enumerate(fused, 1):
            b["id"] = i

        logger.info("ensemble: fused geo=%d + color=%d + bordered=%d (pool=%d) -> %d kept"
                    " (dropped %d small bordered, %d oversized pre-fusion, %d demerged)",
                    len(geo), len(col), len(bor), len(pool), len(fused),
                    dropped_small_bordered, dropped_oversized, demerged)
        return fused

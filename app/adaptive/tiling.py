"""
Tiled detection for DENSE plans.

Why tiling (the dense fix the user asked for):
  A whole-page render makes every tiny stall a microscopic fraction of the image,
  so the detectors' min-area filters reject it and abutting cells fuse -- a dense
  grid comes through as a handful of row-blocks. Cropping the SAME render into
  overlapping tiles makes each stall a healthy fraction of *its tile*, so the
  GEOMETRIC pass traces each cell's own contour and every small booth survives.

Pipeline per tile: run the production EnsembleDetector, drop boxes clipped by an
inner tile seam (the neighbour tile holds them whole), offset back to global
coordinates. Then a global containment-aware NMS + an intersection-over-smaller
dedup collapse the overlap-band duplicates.

This module is ORCHESTRATION only -- the detectors and NMS are the same tuned
production code reused via _detectors. Ported from the validated dense run in
booth_pipeline_docker/pdf_hybrid_pipeline.py.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from _detectors import (EnsembleDetector, ColorDetector, BorderedCellDetector,
                        non_max_suppression)

logger = logging.getLogger(__name__)

_SEAM_EDGE = 4  # px: a box within this of an inner tile margin is "clipped"


def _tile_origins(extent: int, tile: int, step: int) -> List[int]:
    """Tile start offsets so the last tile always reaches `extent`."""
    logger.debug("_tile_origins() called extent=%s tile=%s step=%s", extent, tile, step)
    if extent <= tile:
        return [0]
    xs, x = [], 0
    while True:
        xs.append(x)
        if x + tile >= extent:
            break
        x += step
    return xs


def _poly_open(bo: Dict) -> Optional[List[List[float]]]:
    c = bo.get("coordinates")
    if not c:
        return None
    pts = [list(p) for p in c]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts if len(pts) >= 3 else None


def _offset_booth(bo: Dict, dx: int, dy: int) -> None:
    """Shift a tile-local detection back into full-image coordinates."""
    x, y, w, h = bo["bbox"]
    bo["bbox"] = (x + dx, y + dy, w, h)
    c = bo.get("coordinates")
    if c:
        bo["coordinates"] = [[p[0] + dx, p[1] + dy] for p in c]
    cen = bo.get("centroid")
    if cen:
        bo["centroid"] = (cen[0] + dx, cen[1] + dy)


def _clipped_at_seam(bo: Dict, cw: int, ch: int, gx0: int, gy0: int,
                     W: int, H: int) -> bool:
    """True if the box touches an INNER tile edge (so it is a partial booth the
    neighbouring tile holds whole). Boxes flush against the real image border are
    kept."""
    x, y, w, h = bo["bbox"]
    if x <= _SEAM_EDGE and gx0 > 0:
        return True
    if x + w >= cw - _SEAM_EDGE and gx0 + cw < W:
        return True
    if y <= _SEAM_EDGE and gy0 > 0:
        return True
    if y + h >= ch - _SEAM_EDGE and gy0 + ch < H:
        return True
    return False


def _drop_contained(booths: List[Dict], ios_thresh: float = 0.6) -> List[Dict]:
    """Drop a box when >= ios_thresh of ITS OWN area sits inside a larger kept box.

    Global NMS uses IoU, which is tiny for a small box nested in a big one, so a
    stray sub-cell inside a real booth survives NMS. Intersection-over-smaller
    catches exactly that containment while leaving abutting neighbours (IoS ~ 0)
    alone."""
    logger.debug("_drop_contained() called n_booths=%s ios_thresh=%s",
                 len(booths), ios_thresh)
    def area(b: Dict) -> float:
        x, y, w, h = b["bbox"]
        return float(w) * float(h)

    def ios(a: Dict, b: Dict) -> float:
        ax, ay, aw, ah = a["bbox"]; bx, by, bw, bh = b["bbox"]
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        s = min(area(a), area(b))
        return inter / s if s > 0 else 0.0

    order = sorted(booths, key=area, reverse=True)
    out: List[Dict] = []
    for b in order:
        if any(ios(b, o) >= ios_thresh and area(o) > area(b) for o in out):
            continue
        out.append(b)
    logger.debug("_drop_contained: %s -> %s after containment dedup", len(booths), len(out))
    return out


def detect_tiled(bgr,
                 work_dir: Path,
                 build_detector: Callable[[], EnsembleDetector],
                 tile: int = 1800,
                 overlap: int = 400,
                 iou_threshold: float = 0.4,
                 ios_thresh: float = 0.6,
                 log: Optional[Callable[[str], None]] = None) -> List[Dict]:
    """Run `build_detector()` on overlapping crops of `bgr` and fuse globally.

    `build_detector` is a zero-arg factory so the caller (pipeline) owns the
    detector configuration; we just call it once and reuse the instance across
    tiles. `work_dir` only holds a scratch tile PNG that is removed at the end.
    Returns booths in full-image pixel coordinates.
    """
    logger.debug("detect_tiled() called tile=%s overlap=%s iou_threshold=%s ios_thresh=%s",
                 tile, overlap, iou_threshold, ios_thresh)
    def _log(m: str):
        if log:
            log(m)

    H, W = bgr.shape[:2]
    det = build_detector()
    step = max(1, tile - overlap)
    xs = _tile_origins(W, tile, step)
    ys = _tile_origins(H, tile, step)
    tmp_tile = Path(work_dir) / "_tile.png"
    total = len(xs) * len(ys)
    logger.info("detect_tiled: %sx%s = %s tiles (tile=%s overlap=%s step=%s) image=%sx%s",
                len(xs), len(ys), total, tile, overlap, step, W, H)
    _log(f"[tile] {len(xs)}x{len(ys)} = {total} tiles "
         f"(tile={tile}, overlap={overlap}, step={step})")

    pool: List[Dict] = []
    n_tiles = n_raw = 0
    for gy0 in ys:
        for gx0 in xs:
            crop = bgr[gy0:min(gy0 + tile, H), gx0:min(gx0 + tile, W)]
            ch, cw = crop.shape[:2]
            cv2.imwrite(str(tmp_tile), crop)
            booths = det.detect(str(tmp_tile))
            n_raw += len(booths)
            n_tiles += 1
            logger.debug("detect_tiled: tile %s/%s at (%s,%s) %sx%s -> %s raw, pool=%s",
                         n_tiles, total, gx0, gy0, cw, ch, len(booths), len(pool))
            for b in booths:
                if _clipped_at_seam(b, cw, ch, gx0, gy0, W, H):
                    continue
                _offset_booth(b, gx0, gy0)
                pool.append(b)
            if n_tiles % 10 == 0 or n_tiles == total:
                _log(f"[tile] {n_tiles}/{total} done, pool={len(pool)}")
    if tmp_tile.exists():
        tmp_tile.unlink()
    logger.info("detect_tiled: %s raw across %s tiles, %s after seam-clip",
                n_raw, n_tiles, len(pool))
    _log(f"[detect] {n_raw} raw across tiles, {len(pool)} after seam-clip")

    nms_in = []
    for b in pool:
        x, y, w, h = b["bbox"]
        entry = {"bbox": [x, y, x + w, y + h],
                 "score": float(b.get("score", 1.0)), "_ref": b}
        poly = _poly_open(b)
        if poly is not None:
            entry["poly"] = poly
        nms_in.append(entry)
    logger.debug("detect_tiled: running global NMS on %s pooled boxes (iou=%s)",
                 len(nms_in), iou_threshold)
    kept = [k["_ref"] for k in non_max_suppression(nms_in, iou_threshold=iou_threshold)]
    kept = _drop_contained(kept, ios_thresh)
    logger.info("detect_tiled: %s -> %s booths after global NMS + dedup", len(pool), len(kept))
    _log(f"[merge] {len(pool)} -> {len(kept)} booths after global NMS + dedup")
    return kept


# --------------------------------------------------------------------------- #
# Big-region pass: recover halls / stages / standalone big booths that tiling
# structurally drops (any box wider/taller than the overlap is seam-clipped in
# EVERY tile), then arbitrate each against the crops by coverage.
# --------------------------------------------------------------------------- #
def _area(b: Dict) -> float:
    x, y, w, h = b["bbox"]
    return float(w) * float(h)


def _overlap_area(bbox: Tuple[float, float, float, float],
                  r: Tuple[float, float, float, float]) -> float:
    x, y, w, h = bbox
    x0, y0, x1, y1 = r
    ix = max(0.0, min(x + w, x1) - max(x, x0))
    iy = max(0.0, min(y + h, y1) - max(y, y0))
    return ix * iy


def _scale_booth(b: Dict, f: float) -> None:
    """Scale a booth's geometry by factor f (downscaled-page px -> full px)."""
    x, y, w, h = b["bbox"]
    b["bbox"] = (int(round(x * f)), int(round(y * f)),
                 int(round(w * f)), int(round(h * f)))
    c = b.get("coordinates")
    if c:
        b["coordinates"] = [[p[0] * f, p[1] * f] for p in c]
    cen = b.get("centroid")
    if cen:
        b["centroid"] = (cen[0] * f, cen[1] * f)


def _distinct_label_cells(text_items: List[Dict], B: Dict, cell: int) -> int:
    """Count DISTINCT coarse-grid cells (size `cell` px) holding a text label
    inside B. Counting raw rects is unreliable: the text layer splits one big
    title ('MAIN STAGE A50') into many overlapping rects, which would falsely
    flag a standalone region as a packed hall. Snapping to a booth-sized grid
    collapses one title to ~1 cell while a real booth grid keeps dozens."""
    logger.debug("_distinct_label_cells() called n_text_items=%s cell=%s",
                 len(text_items), cell)
    x, y, w, h = B["bbox"]
    cells = set()
    for ti in text_items:
        cx, cy = ti["center_px"]
        if x <= cx <= x + w and y <= cy <= y + h:
            cells.add((int(cx // cell), int(cy // cell)))
    return len(cells)


def detect_big_regions(bgr,
                       work_dir: Path,
                       neutral_gray: Optional[Tuple[int, int]],
                       max_edge: int,
                       big_min_area_frac: float,
                       big_max_area_frac: float,
                       big_min_side_px: int,
                       log: Optional[Callable[[str], None]] = None) -> List[Dict]:
    """FULL-IMAGE pass for BIG regions (halls, stages, standalone big booths) the
    tiled pass structurally cannot see: any box wider/taller than the tile
    overlap is clipped by a seam in every tile and dropped.

    Runs colour then bordered SEQUENTIALLY (not the parallel ensemble) on a
    DOWNSCALED copy of the whole page -- both choices are validated:
      * memory -- the parallel colour+bordered ensemble OOMs on a ~190MP page;
        capping the long edge at max_edge (~half) and gc'ing between detectors
        holds the peak well inside budget.
      * recall -- downsampling anti-aliases a DASHED border (a rotated
        'MAIN STAGE' diamond) into a connected line so BorderedCellDetector
        finally closes its contour.

    Keeps only genuinely BIG boxes (min side >= big_min_side_px in FULL px); the
    tiled pass already owns anything smaller. NMS collapses colour+bordered both
    firing on the same hall. The standalone-vs-container verdict is made later by
    merge_big_regions. `work_dir` holds a scratch PNG removed at the end."""
    logger.debug("detect_big_regions() called max_edge=%s big_min_area_frac=%s "
                 "big_max_area_frac=%s big_min_side_px=%s neutral_gray=%s",
                 max_edge, big_min_area_frac, big_max_area_frac, big_min_side_px,
                 neutral_gray)
    def _log(m: str):
        if log:
            log(m)

    H, W = bgr.shape[:2]
    page_area = float(W) * float(H)
    long_edge = max(H, W)
    sf = (max_edge / long_edge) if (max_edge and long_edge > max_edge) else 1.0
    if sf < 1.0:
        logger.debug("detect_big_regions: downscaling page %sx%s by sf=%s", W, H, sf)
        small_img = cv2.resize(bgr, (int(round(W * sf)), int(round(H * sf))),
                               interpolation=cv2.INTER_AREA)
    else:
        logger.debug("detect_big_regions: no downscale needed (sf=%s)", sf)
        small_img = bgr.copy()
    tmp = Path(work_dir) / "_bigpass.png"
    cv2.imwrite(str(tmp), small_img)
    del small_img
    gc.collect()
    _log(f"[big] full-image pass on downscaled page (sf={sf:.3f}, "
         f"max_edge={max_edge}, sequential colour->bordered)")

    inv = 1.0 / sf
    raw: List[Dict] = []
    # SEQUENTIAL build->detect->free, one detector at a time, so their working
    # sets never coexist (memory) and the order is explicit.
    builders = [
        ("color", lambda: ColorDetector(run_ocr=False, neutral_gray=neutral_gray,
                                        min_area_frac=big_min_area_frac,
                                        max_area_frac=big_max_area_frac)),
        ("bordered", lambda: BorderedCellDetector(run_ocr=False,
                                                  min_area_frac=big_min_area_frac,
                                                  max_area_frac=big_max_area_frac)),
    ]
    for name, make in builders:
        logger.info("detect_big_regions: running %s detector on downscaled page", name)
        det = make()
        res = det.detect(str(tmp))
        logger.debug("detect_big_regions: %s returned %s raw boxes", name, len(res))
        kept_here = 0
        for b in res:
            if sf < 1.0:
                _scale_booth(b, inv)
            x, y, w, h = b["bbox"]
            if min(w, h) < big_min_side_px:
                continue                   # tiling already owns boxes this small
            if _area(b) > big_max_area_frac * page_area:
                continue                   # the whole-floor outline, not a booth
            b["source"] = "bigregion"
            b["_big_det"] = name
            raw.append(b)
            kept_here += 1
        _log(f"[big]   {name:8s} {len(res)} raw -> {kept_here} big")
        del det, res
        gc.collect()
    if tmp.exists():
        tmp.unlink()

    # NMS so colour+bordered firing on the same hall collapse to one box
    nms_in = []
    for b in raw:
        x, y, w, h = b["bbox"]
        entry = {"bbox": [x, y, x + w, y + h],
                 "score": float(b.get("score", 1.0)), "_ref": b}
        poly = _poly_open(b)
        if poly is not None:
            entry["poly"] = poly
        nms_in.append(entry)
    logger.debug("detect_big_regions: running NMS on %s big raw boxes", len(raw))
    kept = [k["_ref"] for k in non_max_suppression(nms_in, iou_threshold=0.4)]
    logger.info("detect_big_regions: %s raw -> %s big-region candidates "
                "(sf=%.3f, min_side=%spx)", len(raw), len(kept), sf, big_min_side_px)
    _log(f"[big] {len(raw)} raw -> {len(kept)} big-region candidates "
         f"(sf={sf:.3f}, min_side={big_min_side_px}px)")
    return kept


def merge_big_regions(small: List[Dict], big: List[Dict], text_items: List[Dict],
                      coverage_thresh: float = 0.15, max_inner_labels: int = 6,
                      label_cell_px: int = 250,
                      drop_inner_when_standalone: bool = True,
                      log: Optional[Callable[[str], None]] = None) -> List[Dict]:
    """Arbitrate each full-image big box against the crops:

        coverage = (area of crop booths overlapping B) / area(B)
                 == how much of the big box the crops already describe.

      * coverage >= coverage_thresh (OR B holds > max_inner_labels DISTINCT label
        cells) -> B is a HALL CONTAINER wrapping a dense grid -> DROP B, keep the
        crops.
      * otherwise -> B is mostly empty of crops -> a STANDALONE big feature
        (Main Stage, court, BAR) -> KEEP B; with drop_inner_when_standalone, drop
        the few stray crops inside it too ('keep the big box only').

    coverage_thresh 0.15: measured halls fill 0.18-0.50 (aisles/gaps) and some
    carry no text layer, so 0.20+ would leave a 0.18 hall KEPT as one giant box
    swallowing its grid; the distinct-label-cell test catches densely-labelled
    grids that coverage alone misses."""
    logger.debug("merge_big_regions() called n_small=%s n_big=%s n_text_items=%s "
                 "coverage_thresh=%s max_inner_labels=%s",
                 len(small), len(big), len(text_items), coverage_thresh, max_inner_labels)
    def _log(m: str):
        if log:
            log(m)

    kept_big = 0
    keep_big: List[Dict] = []
    drop_inner = set()                       # id() of crops a standalone box ate
    for B in big:
        Ba = _area(B)
        if Ba <= 0:
            continue
        bx, by, bw, bh = B["bbox"]
        Br = (bx, by, bx + bw, by + bh)
        inner_area = 0.0
        overlapping_s = []
        for s in small:
            ix_area = _overlap_area(s["bbox"], Br)
            if ix_area > 0:
                overlapping_s.append(s)
                inner_area += ix_area
        coverage = inner_area / Ba
        n_labels = _distinct_label_cells(text_items, B, label_cell_px)
        is_container = coverage >= coverage_thresh or n_labels > max_inner_labels
        B["_coverage"] = round(coverage, 3)
        B["_n_inner"] = len(overlapping_s)
        B["_n_inner_labels"] = n_labels
        logger.debug("merge_big_regions: candidate coverage=%.3f n_inner=%s "
                     "n_labels=%s is_container=%s", coverage, len(overlapping_s),
                     n_labels, is_container)
        if is_container:
            B["region_role"] = "container_dropped"
            continue
        B["region_role"] = "big"
        for s in overlapping_s:
            if drop_inner_when_standalone:
                s["region_role"] = "inside_big_dropped"
                drop_inner.add(id(s))
            else:
                s["region_role"] = "inside_big"
        keep_big.append(B)
        kept_big += 1
    out = [s for s in small if id(s) not in drop_inner] + keep_big
    logger.info("merge_big_regions: %s/%s big candidates kept standalone "
                "(%s dropped as containers); %s inner crops absorbed; %s total out",
                kept_big, len(big), len(big) - kept_big, len(drop_inner), len(out))
    _log(f"[big] {kept_big}/{len(big)} big candidates kept as standalone booths "
         f"({len(big) - kept_big} dropped as hall containers); "
         f"{len(drop_inner)} inner crops absorbed")
    return out

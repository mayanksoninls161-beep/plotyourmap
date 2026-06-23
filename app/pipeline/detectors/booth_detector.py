"""
booth_detector.py – Unified exhibition-booth detector
=====================================================

A single, coherent detector that replaces the bt_2/bt_3/bt_5/bt_6 stack.

Core idea
---------
Booths on every expo / floor-plan map are RECTANGULAR CELLS that are either
(a) filled with a distinct colour, or (b) white but enclosed by dark borders.
The robust, map-agnostic way to find them is line-based CELL EXTRACTION
(the classic table-recognition technique):

  1. Build a "strong structure" mask = booth border lines.  A border pixel is
     one that is notably DARKER than its local surround (contrast > margin) OR
     a strong colour edge.  This single test is what separates a real booth
     border from the faint decorative background grid (whose lines have
     contrast ~10-25, far below booth borders at >60).

  2. Keep only LONG horizontal + vertical runs (morphological opening).  Text,
     icons and dimension labels have no long straight runs, so they vanish.

  3. The grid of lines partitions the page into CELLS (connected components of
     the inverse).  Each interior cell is a booth candidate – including every
     small cell in a dense grid, with no fragile "subdivide-after-merge" step.

  4. Validate each cell: reject the page background, the walkway, the empty
     reference-grid squares, and anything the wrong size / shape.  A cell is a
     booth iff it is colour-filled OR it is enclosed by enough strong border.

This directly fixes the four reported failures:
  • asiatech / bharat dense grids   → each cell detected individually
  • gmdc / page_1 background grid    → rejected (faint borders, bg interior)
  • automechanika merged booths      → thin colour-edge separators split them
  • automechanika tilted halls       → optional oriented pass (see TILT)

Usage:
  python booth_detector.py <image> [-o out_dir] [--no-tilt]
"""

from __future__ import annotations
import json, logging, math, os, re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np
try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parameters (all relative to image size or local statistics)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Params:
    # working resolution: scale so the long side lands in this band
    target_long_side: int = 2300
    max_upscale:      float = 3.0

    # background flood
    dark_barrier:     int   = 130    # V below this = dark border (flood barrier)
    # adaptive tight flood (recovers faint light-grey booths on a light floor)
    tight_flood_pagebg: float = 0.68 # loose-flood coverage above which to switch
    tight_flood_sat:    float = 0.18 # ...but only if saturated content is below this
    tight_flood_tol:    int   = 12   # per-channel colour tolerance for tight flood

    # internal cut lines (split touching booths). Lower contrast than a
    # barrier – safe because background (incl. reference grid) is already gone.
    cut_contrast:     int   = 22     # px this much darker than surround = a cut
    bright_contrast:  int   = 255    # disabled globally (over-fragments tiny booths)
    line_len_frac:    float = 0.020  # coarse cut-line length; fine pass is 0.55x
    color_edge_sat:   int   = 55     # saturation gradient above this = colour wall
    close_gap:        int   = 3      # px gap to close at line junctions

    # gray walkway (unsaturated mid-value corridor)
    gray_walk_v_lo:   int   = 95
    gray_walk_v_hi:   int   = 205
    gray_walk_frac:   float = 0.015  # min image fraction to treat gray as walkway

    # cell geometry filters (fractions of WORKING image area)
    min_area_frac:    float = 8e-5   # ~ a small 3x3m stall
    max_area_frac:    float = 0.06   # whole zones rejected
    min_side_px:      int   = 12
    min_rect_score:   float = 0.60   # cell fill / bbox area
    min_aspect:       float = 0.14   # short/long

    # booth-vs-background test
    bg_value_margin:  int   = 18     # interior brighter-than this below bg = empty
    bg_sat_max:       int   = 22     # interior saturation below this = "white"
    perim_strong_min: float = 0.18   # min fraction of perimeter on a strong border

    # walkway
    walkway_area_frac: float = 0.012 # a colour CC bigger than this may be walkway
    walkway_hue_tol:   int   = 10

    # tilt
    enable_tilt:      bool  = True
    tilt_min_deg:     float = 8.0
    tilt_max_deg:     float = 82.0

    # localized sub-booth splitting (split inner stalls inside a detected booth)
    enable_subdivide:    bool  = True
    subdiv_min_area_frac: float = 1.2e-3  # only split booths bigger than this
    subdiv_min_side:      int   = 28
    subdiv_contrast:      int   = 16     # internal divider darkness vs booth fill
    subdiv_span:          float = 0.45   # divider min length as frac of booth side
    subdiv_bridge:        int   = 9      # close gaps to bridge dashed dividers

    # bright-cell path (booths brighter than the floor, e.g. white-on-grey)
    enable_bright:        bool  = True
    bright_cell_contrast: int   = 10     # how much brighter than local floor


@dataclass
class Booth:
    id: int
    bbox: Tuple[int, int, int, int]
    area: float
    centroid: Tuple[float, float]
    source: str
    name: Optional[str] = field(default=None)
    coords: Optional[list] = field(default=None)   # oriented quad (orig scale) if tilted

    def to_dict(self):
        logger.debug("to_dict() called id=%s", self.id)
        x, y, w, h = self.bbox
        if self.coords:
            coords = [list(p) for p in self.coords]
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]
        else:
            coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        d = {"id": self.id, "coordinates": coords, "area": self.area,
             "centroid": list(self.centroid), "source": self.source}
        if self.name is not None:
            d["name"] = self.name
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def _scale(bgr: np.ndarray, p: Params) -> Tuple[np.ndarray, float]:
    logger.debug("_scale() called shape=%s", bgr.shape)
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side < p.target_long_side:
        s = min(p.max_upscale, p.target_long_side / long_side)
        interp = cv2.INTER_CUBIC
        logger.debug("_scale: upscaling long_side=%d by s=%.3f", long_side, s)
    elif long_side > p.target_long_side * 1.6:
        s = p.target_long_side / long_side          # downscale huge images
        interp = cv2.INTER_AREA
        logger.debug("_scale: downscaling long_side=%d by s=%.3f", long_side, s)
    else:
        logger.debug("_scale: no resize needed (long_side=%d)", long_side)
        return bgr, 1.0
    return cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=interp), s


def _page_background(bgr: np.ndarray, walkway: np.ndarray, p: "Params"):
    """Page background = light region reachable from the image border by a flood
    fill whose ONLY barriers are (a) dark booth borders and (b) the walkway.

    Faint decorative reference-grid lines (V ~190-220) are intentionally NOT
    barriers, so an empty reference grid floods entirely as background and is
    excluded – this is what kills the gmdc / page_1 grid false positives.
    Returns (page_bg_mask, bg_value)."""
    logger.debug("_page_background() called shape=%s", bgr.shape)
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    unsat_V = V[S < 25]
    bg_val = int(np.percentile(unsat_V, 80)) if unsat_V.size > 1000 else 245
    bg_val = int(np.clip(bg_val, 200, 255))
    logger.debug("_page_background: bg_val=%d", bg_val)

    dark = (V < p.dark_barrier).astype(np.uint8) * 255
    barriers = cv2.bitwise_or(dark, walkway)
    barriers = cv2.dilate(barriers, np.ones((3, 3), np.uint8), 1)

    # ---- loose flood (default): light-ish, low-sat, not a barrier ------------
    trav = ((V >= bg_val - 60) & (S < 40)).astype(np.uint8) * 255
    trav = cv2.bitwise_and(trav, cv2.bitwise_not(barriers))
    flood = trav.copy()
    ff = np.zeros((H + 2, W + 2), np.uint8)
    step = max(1, min(H, W) // 120)
    for x in range(0, W, step):
        for y in (0, H - 1):
            if trav[y, x] and flood[y, x] == 255:
                cv2.floodFill(flood, ff, (x, y), 128)
    for y in range(0, H, step):
        for x in (0, W - 1):
            if trav[y, x] and flood[y, x] == 255:
                cv2.floodFill(flood, ff, (x, y), 128)
    loose = (flood == 128).astype(np.uint8) * 255
    logger.debug("_page_background: loose flood covers %.3f of page", (loose > 0).mean())

    # ---- adaptive tight flood ------------------------------------------------
    # When the loose flood swallows most of the page AND there is little
    # saturated content, the booths are faint light-grey cells distinguished
    # from the floor only by a small shade gap (e.g. AWS expo maps). A colour-
    # TOLERANCE flood stops at that shade gap so the booths survive as
    # foreground. (Tiny reference-grid cells are still dropped later by area.)
    sat_frac = float((S > 40).mean())
    if (loose > 0).mean() > p.tight_flood_pagebg and sat_frac < p.tight_flood_sat:
        logger.debug("_page_background: using adaptive tight flood (sat_frac=%.3f)", sat_frac)
        ff2 = np.zeros((H + 2, W + 2), np.uint8)
        ff2[1:-1, 1:-1] = (barriers > 0).astype(np.uint8)
        img = bgr.copy()
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        t = p.tight_flood_tol
        for x in range(0, W, step):
            for y in (0, H - 1):
                if ff2[y + 1, x + 1] == 0 and S[y, x] < 40 and V[y, x] >= bg_val - 50:
                    cv2.floodFill(img, ff2, (x, y), 0, (t,) * 3, (t,) * 3, flags)
        for y in range(0, H, step):
            for x in (0, W - 1):
                if ff2[y + 1, x + 1] == 0 and S[y, x] < 40 and V[y, x] >= bg_val - 50:
                    cv2.floodFill(img, ff2, (x, y), 0, (t,) * 3, (t,) * 3, flags)
        logger.debug("_page_background: returning tight-flood page background")
        return (ff2[1:-1, 1:-1] == 255).astype(np.uint8) * 255, bg_val

    logger.debug("_page_background: returning loose-flood page background")
    return loose, bg_val


# ─────────────────────────────────────────────────────────────────────────────
# Strong structure (booth borders, not the faint reference grid)
# ─────────────────────────────────────────────────────────────────────────────
def _cut_lines(bgr: np.ndarray, p: Params, line_len=None) -> np.ndarray:
    """Long horizontal + vertical dividers used to split touching booths.
    Uses a LOW contrast threshold (catches faint internal separators) – this is
    safe because the page background and reference grid are removed beforehand,
    so faint lines here only ever split genuine booth blocks."""
    logger.debug("_cut_lines() called line_len=%s", line_len)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    darker = cv2.subtract(bg, gray)
    dark_mask = (darker > p.cut_contrast).astype(np.uint8) * 255

    # bright lines: gaps that are BRIGHTER than their (coloured) surround –
    # i.e. thin white aisles between adjacent same-colour booths.  Harmless on
    # white-on-white booths (no contrast there).
    brighter = cv2.subtract(gray, bg)
    bright_mask = (brighter > p.bright_contrast).astype(np.uint8) * 255

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    sob = cv2.magnitude(cv2.Sobel(S, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(S, cv2.CV_32F, 0, 1, ksize=3))
    col_edge = (sob > p.color_edge_sat * 4).astype(np.uint8) * 255
    raw = cv2.bitwise_or(cv2.bitwise_or(dark_mask, bright_mask), col_edge)

    H, W = gray.shape
    llf = line_len if line_len is not None else p.line_len_frac
    L = max(8, int(llf * min(H, W)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    horiz = cv2.morphologyEx(raw, cv2.MORPH_OPEN, hk)
    vert  = cv2.morphologyEx(raw, cv2.MORPH_OPEN, vk)
    grid = cv2.bitwise_or(horiz, vert)
    logger.debug("_cut_lines: cut grid covers %.4f of image (L=%d)", (grid > 0).mean(), L)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.close_gap, p.close_gap))
    return cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k, iterations=1)


# ─────────────────────────────────────────────────────────────────────────────
# Walkway (large dominant-hue saturated region) – excluded from booths
# ─────────────────────────────────────────────────────────────────────────────
def _gray_walkway(bgr: np.ndarray, p: Params) -> np.ndarray:
    """Largest connected unsaturated mid-value (gray corridor) region."""
    logger.debug("_gray_walkway() called shape=%s", bgr.shape)
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    gray = ((S < 30) & (V >= p.gray_walk_v_lo) & (V < p.gray_walk_v_hi)).astype(np.uint8) * 255
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(gray, 8)
    if n <= 1:
        logger.debug("_gray_walkway: no gray components found")
        return np.zeros((H, W), np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < p.gray_walk_frac * H * W:
        logger.debug("_gray_walkway: largest gray region too small, no walkway")
        return np.zeros((H, W), np.uint8)
    logger.debug("_gray_walkway: walkway found area=%d", int(stats[big, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8) * 255


def _color_walkway(bgr: np.ndarray, p: Params) -> np.ndarray:
    logger.debug("_color_walkway() called shape=%s", bgr.shape)
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hh, S = hsv[..., 0].astype(np.int16), hsv[..., 1]
    sat = S > 50
    if sat.mean() < 0.03:
        logger.debug("_color_walkway: too little saturated content (%.4f), no walkway", float(sat.mean()))
        return np.zeros((H, W), np.uint8)
    hues = Hh[sat].clip(0, 179)
    hist = np.bincount(hues, minlength=180).astype(np.float32)
    k = 5
    ext = np.concatenate([hist[-k:], hist, hist[:k]])
    sm = np.convolve(ext, np.ones(2 * k + 1) / (2 * k + 1), "valid")
    dom = int(np.argmax(sm))
    logger.debug("_color_walkway: dominant hue=%d", dom)
    diff = np.minimum(np.abs(Hh - dom), 180 - np.abs(Hh - dom))
    cand = ((diff <= p.walkway_hue_tol) & (S > 45)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    if n <= 1:
        logger.debug("_color_walkway: no candidate components found")
        return np.zeros((H, W), np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < p.walkway_area_frac * H * W:
        logger.debug("_color_walkway: largest color region too small, no walkway")
        return np.zeros((H, W), np.uint8)
    logger.debug("_color_walkway: walkway found area=%d", int(stats[big, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8) * 255


# ─────────────────────────────────────────────────────────────────────────────
# Cell extraction + booth validation
# ─────────────────────────────────────────────────────────────────────────────
def _extract_booths(bgr, p: Params, source="axis", line_len=None):
    """Booths = (NOT page_bg) AND (NOT walkway) AND (NOT dark border), then cut
    along internal dividers and connected-component label."""
    logger.debug("_extract_booths() called source=%s line_len=%s", source, line_len)
    H, W = bgr.shape[:2]
    total = H * W
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]

    color_wk = np.zeros((H, W), np.uint8)   # disabled: a dominant booth colour
    # (e.g. orange stalls) was being mis-detected as walkway. A real colour
    # walkway forms one over-size region that the max-area filter rejects anyway.
    gray_wk  = _gray_walkway(bgr, p)
    walkway  = cv2.bitwise_or(color_wk, gray_wk)
    page_bg, bg_val = _page_background(bgr, walkway, p)
    cuts = _cut_lines(bgr, p, line_len=line_len)
    dark = (V < p.dark_barrier).astype(np.uint8) * 255

    # foreground booth pixels
    excluded = cv2.bitwise_or(cv2.bitwise_or(page_bg, walkway), dark)
    fg = cv2.bitwise_not(excluded)
    # fill small text/icon holes so a booth interior is one solid blob
    fg = _fill_holes(fg, 6e-4 * total)
    # cut along internal dividers, then erode to break 1px bridges
    fg = cv2.bitwise_and(fg, cv2.bitwise_not(cuts))
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8), 1)
    logger.debug("_extract_booths: foreground mask built, %.4f of image", (fg > 0).mean())

    min_area = p.min_area_frac * total
    max_area = p.max_area_frac * total
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 4)
    logger.debug("_extract_booths: %d candidate components", n - 1)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if min(w, h) < p.min_side_px:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        comp = (lab[y:y + h, x:x + w] == i)
        rect = area / float(w * h)
        if rect < p.min_rect_score:
            continue
        if min(w, h) / max(w, h) < p.min_aspect:
            continue
        out.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": float(area),
            "centroid": (float(cent[i][0]), float(cent[i][1])),
            "source": source,
        })
    logger.debug("_extract_booths: kept %d booths (source=%s)", len(out), source)
    return out, bg_val


def _fill_holes(mask: np.ndarray, max_hole: float) -> np.ndarray:
    logger.debug("_fill_holes() called max_hole=%.1f", max_hole)
    H, W = mask.shape
    inv = cv2.bitwise_not(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = mask.copy()
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if a <= max_hole:
            out[lab == i] = 255
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Tilt handling (oriented pass)
# ─────────────────────────────────────────────────────────────────────────────
def _dominant_tilt(strong: np.ndarray, p: Params) -> Optional[float]:
    logger.debug("_dominant_tilt() called shape=%s", strong.shape)
    H, W = strong.shape
    lines = cv2.HoughLinesP(strong, 1, np.pi / 180,
                            threshold=int(0.10 * min(H, W)),
                            minLineLength=int(0.05 * min(H, W)), maxLineGap=8)
    if lines is None:
        logger.debug("_dominant_tilt: no Hough lines found")
        return None
    angs = []
    for l in lines[:, 0]:
        a = math.degrees(math.atan2(l[3] - l[1], l[2] - l[0])) % 180
        if a > 90:
            a -= 180
        angs.append(a)
    angs = np.array(angs)
    tilt = angs[(np.abs(angs) > p.tilt_min_deg) & (np.abs(angs) < p.tilt_max_deg)]
    logger.debug("_dominant_tilt: %d lines, %d in tilt band", len(angs), len(tilt))
    if len(tilt) < max(20, 0.10 * len(angs)):
        logger.debug("_dominant_tilt: too few tilted lines, no dominant tilt")
        return None
    hist, edges = np.histogram(tilt, bins=np.arange(-90, 91, 3))
    b = int(np.argmax(hist))
    if hist[b] < 20:
        logger.debug("_dominant_tilt: dominant tilt bin too weak (%d), no tilt", int(hist[b]))
        return None
    tilt_deg = float((edges[b] + edges[b + 1]) / 2)
    logger.debug("_dominant_tilt: dominant tilt=%.1f deg", tilt_deg)
    return tilt_deg


def _rotate(img, deg):
    logger.debug("_rotate() called deg=%s", deg)
    H, W = img.shape[:2]
    c = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(c, deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,
                          borderValue=(255, 255, 255)), M


# ─────────────────────────────────────────────────────────────────────────────
# Dedup
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    logger.debug("_iou() called")
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = aw * ah + bw * bh - inter
    return inter / u if u else 0.0


def _dedupe(items, iou_t=0.40):
    logger.debug("_dedupe() called n_items=%d iou_t=%.2f", len(items), iou_t)
    items = sorted(items, key=lambda c: -c["area"])
    kept = []
    for it in items:
        if any(_iou(it["bbox"], k["bbox"]) > iou_t for k in kept):
            continue
        kept.append(it)
    logger.debug("_dedupe: kept %d of %d", len(kept), len(items))
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def _ios(small, large):
    logger.debug("_ios() called")
    ax, ay, aw, ah = small; bx, by, bw, bh = large
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    s = aw * ah
    return inter / s if s else 0.0


def _dedupe_prefer_fine(items, iou_t=0.55, cover_t=0.80):
    """Keep finer (smaller) boxes first; drop a larger box if most of it is
    already covered by smaller boxes, and drop near-duplicates by IoU."""
    logger.debug("_dedupe_prefer_fine() called n_items=%d iou_t=%.2f cover_t=%.2f",
                 len(items), iou_t, cover_t)
    items = sorted(items, key=lambda c: c["area"])      # small first
    kept = []
    for it in items:
        dup = False
        for k in kept:
            if _iou(it["bbox"], k["bbox"]) > iou_t:
                dup = True
                break
        if dup:
            continue
        kept.append(it)
    logger.debug("_dedupe_prefer_fine: %d kept after IoU pass", len(kept))
    # second pass: drop big boxes mostly tiled by already-kept smaller boxes
    final = []
    kept_sorted = sorted(kept, key=lambda c: c["area"])
    for i, it in enumerate(kept_sorted):
        smaller = [k for k in kept_sorted if k["area"] < it["area"] * 0.9]
        # fraction of `it` covered by union of smaller boxes (approx via grid)
        x, y, w, h = it["bbox"]
        if smaller and w * h > 0:
            acc = np.zeros((max(1, h // 4), max(1, w // 4)), np.uint8)
            for s in smaller:
                sx, sy, sw, sh = s["bbox"]
                ix1, iy1 = max(x, sx), max(y, sy)
                ix2, iy2 = min(x + w, sx + sw), min(y + h, sy + sh)
                if ix2 > ix1 and iy2 > iy1:
                    acc[(iy1 - y) // 4:(iy2 - y) // 4, (ix1 - x) // 4:(ix2 - x) // 4] = 1
            if acc.mean() > cover_t:
                continue
        final.append(it)
    logger.debug("_dedupe_prefer_fine: %d kept after coverage pass", len(final))
    return final


def _subdivide(bgr, cands, p: Params):
    """Localized splitting: for each detected booth, look for STRONG internal
    H/V dividers *inside that booth only* and, if they cleanly partition it into
    >=2 booth-sized sub-cells, replace the booth with its sub-cells.  Because the
    search is confined to a booth's own crop, it never touches walkways, empty
    space, or the page grid – so it can't over-fragment the map."""
    logger.debug("_subdivide() called n_cands=%d", len(cands))
    H, W = bgr.shape[:2]
    total = H * W
    min_area = p.min_area_frac * total
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for c in cands:
        x, y, w, h = c["bbox"]
        # only attempt to split booths comfortably larger than the floor size
        if w * h < p.subdiv_min_area_frac * total or min(w, h) < p.subdiv_min_side:
            out.append(c)
            continue
        pad = 0
        sub = gray[y + pad:y + h - pad, x + pad:x + w - pad]
        if sub.size < 100:
            out.append(c)
            continue
        sh, sw = sub.shape
        bg = cv2.medianBlur(sub, max(11, (min(sh, sw) // 4) | 1))
        dark = cv2.subtract(bg, sub)
        lines = (dark > p.subdiv_contrast).astype(np.uint8) * 255
        Lh = max(8, int(p.subdiv_span * sw))   # divider spans most of the booth
        Lv = max(8, int(p.subdiv_span * sh))
        # bridge DASHED/dotted dividers first (close along the line direction),
        # then keep only long straight runs.
        bridge = p.subdiv_bridge
        hc = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)))
        vc = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge)))
        horiz = cv2.morphologyEx(hc, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (Lh, 1)))
        vert  = cv2.morphologyEx(vc, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (1, Lv)))
        grid = cv2.bitwise_or(horiz, vert)
        if (grid > 0).mean() < 1e-3:
            out.append(c)
            continue
        cells = cv2.bitwise_not(grid)
        cells = cv2.erode(cells, np.ones((3, 3), np.uint8), 1)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(cells, 4)
        pieces = []
        for i in range(1, n):
            cx, cy, cw, ch, ca = stats[i]
            if ca < min_area or ca < 0.12 * (sw * sh):
                continue
            if min(cw, ch) < p.min_side_px:
                continue
            if ca / float(cw * ch) < p.min_rect_score:
                continue
            pieces.append({
                "bbox": (int(x + cx), int(y + cy), int(cw), int(ch)),
                "area": float(ca),
                "centroid": (x + cent[i][0], y + cent[i][1]),
                "source": c["source"],
            })
        # accept the split only if it yields >=2 pieces covering most of the booth
        if len(pieces) >= 2 and sum(pp["area"] for pp in pieces) > 0.45 * w * h:
            out.extend(pieces)
        else:
            out.append(c)
    logger.debug("_subdivide: %d booths in -> %d booths out", len(cands), len(out))
    return out


def _bright_cells(bgr, p: Params, line_len=None):
    """Recover booths that are BRIGHTER than their surrounding floor (e.g. white
    stalls on a grey hall floor, often outlined only by faint dashed borders –
    these are invisible to the elimination path because they read as background).
    A cell here is a region brighter than the large-scale local floor, cut by the
    booth borders.  On white-background maps nothing is brighter than the page, so
    this path contributes nothing and is harmless."""
    logger.debug("_bright_cells() called line_len=%s", line_len)
    H, W = bgr.shape[:2]
    total = H * W
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    floor = cv2.medianBlur(gray, max(31, (min(H, W) // 20) | 1))
    bright = (cv2.subtract(gray, floor) > p.bright_cell_contrast).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cuts = _cut_lines(bgr, p, line_len=line_len)
    fg = cv2.bitwise_and(bright, cv2.bitwise_not(cuts))
    fg = _fill_holes(fg, 6e-4 * total)
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8), 1)
    mn, mx = p.min_area_frac * total, p.max_area_frac * total
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 4)
    logger.debug("_bright_cells: %d candidate bright components", n - 1)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < mn or a > mx or min(w, h) < p.min_side_px:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        if a / float(w * h) < p.min_rect_score or min(w, h) / max(w, h) < p.min_aspect:
            continue
        out.append({"bbox": (int(x), int(y), int(w), int(h)), "area": float(a),
                    "centroid": (float(cent[i][0]), float(cent[i][1])), "source": "bright"})
    logger.debug("_bright_cells: kept %d bright booths", len(out))
    return out


def _ocr_booth_name(bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
    """Crop the booth from the image and run OCR to extract its label.
    Returns the cleaned text string, or None if nothing legible is found."""
    logger.debug("_ocr_booth_name() called bbox=%s", bbox)
    if not _TESSERACT_AVAILABLE:
        logger.debug("_ocr_booth_name: tesseract not available")
        return None
    x, y, w, h = bbox
    H, W = bgr.shape[:2]
    pad = max(2, int(min(w, h) * 0.05))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # upscale small crops so Tesseract has enough pixels
    scale = max(1.0, 80.0 / min(gray.shape[:2]))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # adaptive threshold to handle coloured booth backgrounds
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .&-/"
    try:
        text = pytesseract.image_to_string(thresh, config=config)
    except Exception:
        logger.exception("_ocr_booth_name: pytesseract image_to_string failed")
        return None

    # clean: keep only printable, collapse whitespace, strip noise chars
    text = re.sub(r"[^A-Za-z0-9 .&\-/]", " ", text)
    text = " ".join(text.split()).strip()
    # discard single-character results and pure-digit strings shorter than 2 chars
    if len(text) <= 1:
        return None
    return text or None

import concurrent.futures

def extract_labels_with_ocr(bgr: np.ndarray, booths: List[Booth]) -> None:
    """Run Tesseract in parallel across cropped booths to extract labels."""
    logger.debug("extract_labels_with_ocr() called n_booths=%d", len(booths))
    if not _TESSERACT_AVAILABLE:
        logger.debug("extract_labels_with_ocr: tesseract not available, skipping")
        print("Tesseract not available for text extraction.")
        return

    def _process_booth(booth):
        logger.debug("_process_booth() called id=%s", booth.id)
        name = _ocr_booth_name(bgr, booth.bbox)
        if name:
            booth.name = name

    # Process all booths in parallel threads
    logger.debug("extract_labels_with_ocr: running OCR across %d booths in parallel", len(booths))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_process_booth, booths))
    logger.debug("extract_labels_with_ocr: OCR complete")


def detect(image_path: str, p: Optional[Params] = None) -> Tuple[List[Booth], dict]:
    logger.info("detect() called image_path=%s", image_path)
    if p is None:
        p = Params()
    bgr0 = cv2.imread(image_path)
    if bgr0 is None:
        logger.debug("detect: failed to read image %s", image_path)
        raise FileNotFoundError(image_path)

    bgr, scale = _scale(bgr0, p)
    logger.debug("detect: working scale=%.3f shape=%s", scale, bgr.shape)

    # multi-scale axis-aligned extraction: a COARSE pass keeps big booths whole,
    # a FINE pass splits dense grids.  "Prefer finer" dedup keeps the granular
    # split where it exists and the whole booth elsewhere.
    cands = []
    bg_val = 245
    for ll in (p.line_len_frac, p.line_len_frac * 0.55):
        c, bg_val = _extract_booths(bgr, p, "axis", line_len=ll)
        cands += c
    logger.debug("detect: %d candidates after multi-scale axis extraction", len(cands))
    if p.enable_bright:
        bright = _bright_cells(bgr, p, line_len=p.line_len_frac)
        cands += bright
        logger.debug("detect: +%d bright-cell candidates", len(bright))

    # tilted pass
    if p.enable_tilt:
        ang = _dominant_tilt(_cut_lines(bgr, p), p)
        if ang is not None and abs(ang) > p.tilt_min_deg:
            logger.debug("detect: running tilted pass at ang=%.1f deg", ang)
            rot, M = _rotate(bgr, ang)
            rc, _ = _extract_booths(rot, p, "tilt")
            Minv = cv2.invertAffineTransform(M)
            for c in rc:
                x, y, w, h = c["bbox"]
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
                op = (Minv[:, :2] @ pts.T + Minv[:, 2:]).T            # 4 oriented corners
                ox1, oy1, ox2, oy2 = op[:, 0].min(), op[:, 1].min(), op[:, 0].max(), op[:, 1].max()
                c["bbox"] = (int(ox1), int(oy1), int(ox2 - ox1), int(oy2 - oy1))  # AABB envelope
                c["quad"] = [[float(px), float(py)] for px, py in op]  # KEEP orientation (was discarded)
                c["centroid"] = (float(op[:, 0].mean()), float(op[:, 1].mean()))
            cands += rc
            logger.debug("detect: +%d tilted candidates", len(rc))

    cands = _dedupe_prefer_fine(cands)
    logger.debug("detect: %d candidates after dedupe", len(cands))
    if p.enable_subdivide:
        cands = _subdivide(bgr, cands, p)
        cands = _dedupe_prefer_fine(cands)
        logger.debug("detect: %d candidates after subdivide+dedupe", len(cands))

    inv = 1.0 / scale
    booths = []
    for i, c in enumerate(cands, 1):
        x, y, w, h = c["bbox"]
        quad = c.get("quad")
        coords = ([[float(px * inv), float(py * inv)] for px, py in quad]
                  if quad is not None else None)
        booths.append(Booth(i, (int(x * inv), int(y * inv), int(w * inv), int(h * inv)),
                            c["area"] * inv * inv,
                            (c["centroid"][0] * inv, c["centroid"][1] * inv),
                            c["source"], coords=coords))
    logger.info("detect: returning %d booths (scale=%.3f, bg_val=%d)", len(booths), scale, bg_val)
    return booths, {"scale": scale, "bg_val": bg_val}


def visualize(image_path, booths, out_path):
    logger.debug("visualize() called image_path=%s n_booths=%d out_path=%s",
                 image_path, len(booths), out_path)
    bgr = cv2.imread(image_path)
    overlay = bgr.copy()
    rng = np.random.default_rng(7)
    for b in booths:
        x, y, w, h = b.bbox
        col = tuple(int(c) for c in rng.integers(60, 230, size=3))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), col, -1)
    out = cv2.addWeighted(overlay, 0.22, bgr, 0.78, 0)
    thick = max(1, int(round(min(bgr.shape[:2]) / 900)))
    for b in booths:
        x, y, w, h = b.bbox
        edge = (255, 0, 200) if b.source == "tilt" else (20, 220, 30)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 0), thick + 1)
        cv2.rectangle(out, (x, y), (x + w, y + h), edge, thick)
        fs = max(0.32, min(bgr.shape[:2]) / 4000)
        label_text = f"{b.id}: {b.name}" if b.name else str(b.id)
        cv2.putText(out, label_text, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label_text, (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, out)
    logger.debug("visualize: wrote overlay -> %s", out_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="out_unified")
    ap.add_argument("--no-tilt", action="store_true")
    args = ap.parse_args()
    P = Params(enable_tilt=not args.no_tilt)
    booths, meta = detect(args.image, P)
    os.makedirs(args.out, exist_ok=True)
    visualize(args.image, booths, os.path.join(args.out, "subdivided_visualization.png"))
    with open(os.path.join(args.out, "booths.json"), "w") as f:
        json.dump({"count": len(booths), "booths": [b.to_dict() for b in booths]}, f, indent=2)
    print(f"{os.path.basename(args.image)}: {len(booths)} booths  (scale={meta['scale']:.2f})")

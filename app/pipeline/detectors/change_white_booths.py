"""
Recolor exhibition floor-plan booths by category, robust across many
different floor-plan styles (different greens, yellows, and white shades).

Categories
----------
  * GREEN  "Walking Passage" / green booths  -> HUE-based mask (catches sage,
           lime, pure green, pale green, teal). Keys on hue, not fixed RGB,
           so it works regardless of the exact green shade.
  * YELLOW "Sponsor" / catering / gold blocks -> HUE-based mask.
  * WHITE  "Available" booths -> pure-white connected components that do NOT
           touch the image border. The white PAGE BACKGROUND always touches
           the border, so it is excluded; genuine booths (enclosed by borders,
           a grey floor, or colour) are kept. Avoids repainting the page.

Each category has an ON/OFF switch and a target colour at the top.

Usage
-----
    python change_white_booths.py input.jpg output.jpg
"""

import sys
import cv2
import numpy as np


# ====================== CONFIG: edit these ===============================
# Target colours in RGB (normal order). Set DO_* = False to skip a category.

DO_WHITE  = True
TARGET_WHITE_RGB  = (200, 220, 255)   # light periwinkle blue

DO_YELLOW = True
TARGET_YELLOW_RGB = (255, 170, 120)   # warm coral

DO_GREEN  = True
TARGET_GREEN_RGB  = (240, 176, 0)     # orange-gold

# How to find white/available booths:
#   "simple"   - pure-white blobs that don't touch the page edge. Safe default.
#                Use when booths are a DIFFERENT shade from the page/aisles
#                (CloudExpo, GMDC).
#   "enclosed" - cut the white along DARK booth outlines so each cell becomes its
#                own island. Use for white-on-white booths with black borders on a
#                light grid (Bharat Mandapam). Ignores the light grid lines.
#   "edges"    - cut along DETECTED EDGES (borders of any colour, even faint grey).
#                Use when booths are light GREY with thin grey borders, only a
#                little darker than the page (the AWS maps). Don't use on fine-grid
#                CAD floors (SmartCities) - the grid fragments into many cells.
WHITE_METHOD = "enclosed"

# Keep the black labels, dotted borders, and red arrows that sit ON TOP of
# booths. Without this, anti-aliased text gets swallowed by the fill and the
# booth labels vanish. We re-apply every dark "ink" pixel from the original
# image as a final pass, so text/borders/arrows survive in their original colour.
PRESERVE_TEXT = True
TEXT_DARK_MAX = 125                   # original grayscale < this == ink to keep
# =========================================================================

# ---- GREEN detection (HSV hue, OpenCV hue range 0-179) ------------------
GREEN_HUE_LO, GREEN_HUE_HI = 35, 95
GREEN_SAT_MIN, GREEN_VAL_MIN = 25, 60

# ---- YELLOW / gold detection --------------------------------------------
YELLOW_HUE_LO, YELLOW_HUE_HI = 10, 35
YELLOW_SAT_MIN, YELLOW_VAL_MIN = 60, 120

# ---- WHITE detection ----------------------------------------------------
SAT_MAX = 30        # HSV saturation < this -> white-ish
VAL_MIN = 200


def _rgb_to_bgr(rgb):
    r, g, b = rgb
    return np.array((b, g, r), dtype=np.uint8)


def recolor_green(img, out, color_bgr):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((Hh >= GREEN_HUE_LO) & (Hh <= GREEN_HUE_HI) &
            (S >= GREEN_SAT_MIN) & (V >= GREEN_VAL_MIN))
    out[mask] = color_bgr
    print(f"  green:  {int(mask.sum())} px")
    return int(mask.sum())


def recolor_yellow(img, out, color_bgr):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(int)
    Hh, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = ((Hh >= YELLOW_HUE_LO) & (Hh < YELLOW_HUE_HI) &
            (S >= YELLOW_SAT_MIN) & (V >= YELLOW_VAL_MIN))
    out[mask] = color_bgr
    print(f"  yellow: {int(mask.sum())} px")
    return int(mask.sum())


def _ring_non_white_frac(white_mask, x, y, w, h, ring_w=2):
    """Fraction of the 1-2px ring outside the bbox that is NOT white."""
    H, W = white_mask.shape
    x0, y0 = max(0, x - ring_w), max(0, y - ring_w)
    x1, y1 = min(W, x + w + ring_w), min(H, y + h + ring_w)
    parts = []
    if y0 < y:     parts.append(white_mask[y0:y,     x0:x1])
    if y1 > y + h: parts.append(white_mask[y + h:y1, x0:x1])
    if x0 < x:     parts.append(white_mask[y:y + h,  x0:x])
    if x1 > x + w: parts.append(white_mask[y:y + h,  x + w:x1])
    if not parts:
        return 0.0
    ring = np.concatenate([r.ravel() for r in parts])
    return (ring == 0).mean()


def recolor_white(img, out, color_bgr):
    H, W = img.shape[:2]
    img_area = H * W
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < SAT_MAX) & (hsv[:, :, 2] > VAL_MIN)).astype(np.uint8) * 255
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)

    # sanity check: is there at least one large clean white booth?
    biggest_clean = 0
    for i in range(1, n_lbl):
        x, y, w, h, area = stats[i]
        if area > img_area * 0.05:           continue
        if min(w, h) < 15:                   continue
        if area / (w * h) < 0.85:            continue
        if max(w, h) / max(1, min(w, h)) > 6: continue
        biggest_clean = max(biggest_clean, area)
    if biggest_clean < img_area * 0.0010:
        print("  white: none detected (image left unchanged for this category)")
        return 0

    min_dim  = max(8,   int(min(H, W) * 0.007))
    min_area = max(200, int(img_area * 0.00010))
    max_area = int(img_area * 0.012)
    count = 0
    for i in range(1, n_lbl):
        x, y, w, h, area = stats[i]
        if not (min_area <= area <= max_area):                continue
        if min(w, h) < min_dim:                               continue
        if area / (w * h) < 0.78:                             continue
        if max(w, h) / max(1, min(w, h)) > 15:                continue
        if _ring_non_white_frac(white, x, y, w, h, 2) < 0.50: continue
        out[labels == i] = color_bgr
        count += 1
    print(f"  white: recolored {count} booths")
    return count


def recolor(in_path: str, out_path: str) -> None:
    """Recolor green, yellow, and white booths in a floor-plan image and save."""
    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(in_path)
    out = img.copy()
    print(in_path.split("/")[-1])
    if DO_GREEN:
        recolor_green(img, out, _rgb_to_bgr(TARGET_GREEN_RGB))
    if DO_YELLOW:
        recolor_yellow(img, out, _rgb_to_bgr(TARGET_YELLOW_RGB))
    if DO_WHITE:
        recolor_white(img, out, _rgb_to_bgr(TARGET_WHITE_RGB))

    if PRESERVE_TEXT:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ink = gray < TEXT_DARK_MAX
        out[ink] = img[ink]
        print(f"  preserved {int(ink.sum())} ink px (text/borders/arrows)")

    cv2.imwrite(out_path, out)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) >= 2 else "/home/nls34/Documents/POCs/Exhibition_hall/Input_image/techexna2026.png"
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "/home/nls34/Documents/POCs/Exhibition_hall/Input_image/color9/techexna2026.png"
    recolor(in_path, out_path)

"""
Stage 0 -- cheap input characterization.

Looks at a file and decides WHAT KIND of floor plan it is, so the pipeline can
self-configure (see config.py). Everything here is fast + heuristic -- no ML, no
GPU. A VLLM (Qwen) can later REPLACE `characterize()` and emit the same
InputProfile; nothing downstream needs to change.

Three axes that actually change how we should detect:

  1. source    -- pdf_vector | pdf_raster | image
       Decides labeling (PDF text layer vs OCR) and rendering (pdfium scale vs
       reading the raster directly).
  2. booth_fill -- colored | grey | white
       Decides whether the colour pass carries the load (colored), whether we
       open the neutral-grey band (grey), or whether the bordered pass is the
       load-bearing one (white / outline-only booths).
  3. density    -- sparse | normal | dense
       Decides render DPI + whether to tile. Measured scale-invariantly from the
       MEDIAN booth size as a fraction of the page (tiny booths => dense => need
       resolution), backed by a raw count for confidence.

The raw stats are kept on the profile so thresholds can be calibrated against
real plans instead of guessed in the abstract.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Long edge (px) of the small image used only for characterization. Big enough
# that the geometric pass can see cells, small enough to stay ~1-3s.
PROFILE_EDGE = 1700

# --- booth-fill thresholds (fraction of page pixels) ------------------------
COLORED_FRAC_MIN = 0.045    # >= ~4.5% saturated colour => colour-filled booths
GREY_FRAC_MIN = 0.045       # else >= ~4.5% neutral grey => grey-filled booths

# --- density thresholds (median booth area as fraction of page) -------------
DENSE_MED_AREA_MAX = 5e-4   # median booth < 0.05% of page => dense grid
NORMAL_MED_AREA_MAX = 6e-3  # else < 0.6% => normal; bigger => sparse
DENSE_MIN_COUNT = 40        # need at least this many cells to call it dense


@dataclass
class InputProfile:
    path: str
    source: str                 # pdf_vector | pdf_raster | image
    booth_fill: str             # colored | grey | white
    density: str                # sparse | normal | dense
    # raw signals (for calibration / logging)
    page_pt: Optional[Tuple[float, float]] = None   # PDF points (vector only)
    n_pages: int = 1
    n_text_rects: int = 0
    profile_px: Tuple[int, int] = (0, 0)            # (w,h) of profiling image
    colored_frac: float = 0.0
    grey_frac: float = 0.0
    white_frac: float = 0.0
    n_geo: int = 0                                  # geometric cells on profile img
    median_area_frac: float = 0.0
    elapsed_s: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def is_pdf(self) -> bool:
        return self.source.startswith("pdf")

    @property
    def name(self) -> str:
        return f"{self.source}/{self.booth_fill}/{self.density}"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["name"] = self.name        # @property -- asdict() omits it
        d["is_pdf"] = self.is_pdf
        return d


# --------------------------------------------------------------------------- #
def _load_profile_image(path: str) -> Tuple[np.ndarray, str, int, Optional[Tuple[float, float]], int]:
    """Return (bgr_small, source_kind, n_text_rects, page_pt, n_pages).

    For PDFs we render page 0 at a scale capped to PROFILE_EDGE and count text
    rects to tell vector from scanned. For raster files we just read + downscale.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(path)
        n_pages = len(pdf)
        page = pdf[0]
        w_pt, h_pt = page.get_size()
        long_pt = max(w_pt, h_pt)
        scale = PROFILE_EDGE / long_pt if long_pt > 0 else 1.0
        bitmap = page.render(scale=scale)
        rgb = np.asarray(bitmap.to_pil().convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # count non-empty text rects on the page (vector text layer present?)
        tp = page.get_textpage()
        n_rects = 0
        for i in range(tp.count_rects()):
            l, b, r, t = tp.get_rect(i)
            if tp.get_text_bounded(left=l, bottom=b, right=r, top=t).strip():
                n_rects += 1
        tp.close()
        page.close()
        pdf.close()
        source = "pdf_vector" if n_rects >= 3 else "pdf_raster"
        return bgr, source, n_rects, (w_pt, h_pt), n_pages

    # raster image
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = bgr.shape[:2]
    long_px = max(w, h)
    if long_px > PROFILE_EDGE:
        s = PROFILE_EDGE / float(long_px)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return bgr, "image", 0, None, 1


def _colour_fractions(bgr: np.ndarray) -> Tuple[float, float, float]:
    """(colored_frac, grey_frac, white_frac) over all pixels.

    colored = saturated and not too dark/bright; grey = desaturated mid-tone;
    white = bright + desaturated (page background, NOT a booth fill signal)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.int32)
    v = hsv[:, :, 2].astype(np.int32)
    total = float(s.size) or 1.0
    colored = np.count_nonzero((s > 45) & (v > 50) & (v < 250))
    grey = np.count_nonzero((s <= 30) & (v >= 120) & (v <= 210))
    white = np.count_nonzero((s <= 30) & (v > 210))
    return colored / total, grey / total, white / total


def _geometric_density(bgr_small: np.ndarray) -> Tuple[int, float]:
    """Run the geometric pass (fastest, OCR off) on the small image and return
    (n_cells, median_area_frac). Scale-invariant size signal for density."""
    import tempfile, os
    from _detectors import OpenCVDetector
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(tmp, bgr_small)
        boxes = OpenCVDetector(run_ocr=False).detect(tmp)
    except Exception:
        return 0, 0.0
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if not boxes:
        return 0, 0.0
    img_area = float(bgr_small.shape[0] * bgr_small.shape[1]) or 1.0
    hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
    
    areas = []
    for b in boxes:
        x, y, w, h = b["bbox"]
        
        # Check if the center of this box is pure white (likely a background grid cell)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        
        crop_s = hsv[max(0, cy-1):cy+2, max(0, cx-1):cx+2, 1]
        crop_v = hsv[max(0, cy-1):cy+2, max(0, cx-1):cx+2, 2]
        
        # If it's desaturated and bright, it's white (skip it for density calculation)
        if crop_s.size > 0 and np.mean(crop_s) < 25 and np.mean(crop_v) > 220:
            continue
            
        areas.append((w * h) / img_area)
        
    if not areas:
        # Fallback if somehow EVERYTHING was white
        areas = [(b["bbox"][2] * b["bbox"][3]) / img_area for b in boxes]
        
    return len(boxes), float(np.median(areas))


def _classify_fill(colored_frac: float, grey_frac: float) -> str:
    if colored_frac >= COLORED_FRAC_MIN:
        return "colored"
    if grey_frac >= GREY_FRAC_MIN:
        return "grey"
    return "white"


def _classify_density(n_geo: int, median_area_frac: float) -> str:
    if n_geo < 15:
        return "sparse"
    if median_area_frac <= DENSE_MED_AREA_MAX and n_geo >= DENSE_MIN_COUNT:
        return "dense"
    if median_area_frac <= NORMAL_MED_AREA_MAX:
        return "normal"
    return "sparse"


def characterize(path: str) -> InputProfile:
    t0 = time.perf_counter()
    bgr, source, n_rects, page_pt, n_pages = _load_profile_image(path)
    colored_frac, grey_frac, white_frac = _colour_fractions(bgr)
    n_geo, median_area_frac = _geometric_density(bgr)

    booth_fill = _classify_fill(colored_frac, grey_frac)
    density = _classify_density(n_geo, median_area_frac)

    prof = InputProfile(
        path=path,
        source=source,
        booth_fill=booth_fill,
        density=density,
        page_pt=page_pt,
        n_pages=n_pages,
        n_text_rects=n_rects,
        profile_px=(bgr.shape[1], bgr.shape[0]),
        colored_frac=round(colored_frac, 4),
        grey_frac=round(grey_frac, 4),
        white_frac=round(white_frac, 4),
        n_geo=n_geo,
        median_area_frac=round(median_area_frac, 6),
        elapsed_s=round(time.perf_counter() - t0, 2),
    )
    if n_pages > 1:
        prof.notes.append(f"multi-page ({n_pages}); profiling page 0 only")
    return prof


if __name__ == "__main__":
    import sys, json
    for p in sys.argv[1:]:
        prof = characterize(p)
        print(f"\n{Path(p).name}")
        print(f"  -> {prof.name}")
        print(json.dumps(prof.to_dict(), indent=2, default=str))

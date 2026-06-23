"""
Rendering, labeling, tagging, false-positive policy and visualization.

The detector finds geometry; this module decides WHAT each box is and which
boxes are real booths. Both label sources -- the PDF vector text layer and
EasyOCR on rasters -- are normalised into the SAME `text_items` shape and fed
through ONE spatial assignment (`label_booths`) + ONE tagger, so a booth is
tagged identically however its text was obtained.

Lifted (stable, ~unchanged) from the known-good pdf_booth_pipeline reference;
the raster render + EasyOCR `text_items` path is the only new logic.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Text signals -- "this is a real booth" patterns
# --------------------------------------------------------------------------- #
RE_AREA = re.compile(r"\d{1,4}\s*(?:sq\.?\s*m(?:tr)?|sqm|m2|m²)\b", re.I)
RE_BOOTH = re.compile(
    r"\b(?:[A-Z]{1,3}[-\s]?\d{2,4}|\d{1,2}[A-Z]{1,2}[-\s]?\d{1,4})[A-Z]?\b")
RE_COMPANY = re.compile(
    r"\b(?:PVT|LTD|LLP|INC|LLC|EXPORTS?|IMPEX|INDUSTR|ENTERPRIS|"
    r"INTERNATIONAL|TRADERS?|OVERSEAS|LIFECARE|TECHNOLOG)\b", re.I)
RE_FACILITY = re.compile(
    r"\b(?:TOILET|LIFT|DRINKING|WATER|CARGO|SERVICE|FHC|RWP|JC|HUB|LV|"
    r"STAIR|ENTRY|EXIT|GATE|RAMP|PANTRY|FIRE|ELECTRIC|DG|AHU|DUCT|SHAFT)\b", re.I)


def is_boothlike(text: str) -> bool:
    logger.debug("is_boothlike() called len=%s", len(text))
    # Area / company keywords are unambiguous booth signals at any length.
    if RE_AREA.search(text) or RE_COMPANY.search(text):
        return True
    # A bare booth CODE counts only when the label is SHORT. A 20-60 token
    # run-on is text harvested from a fused multi-booth block that merely
    # CONTAINS a code substring -- not one booth's label. Without this guard a
    # whole-row block is mislabelled 'boothlike', and on dense vector plans
    # (OOAK) that tips adaptive into 'strict', which then keeps the few garbled
    # blocks and drops every clean shape.
    return bool(RE_BOOTH.search(text)) and len(text.split()) <= 8


def tag_from_label(label: str) -> str:
    logger.debug("tag_from_label() called len=%s", len(label))
    if not label:
        return "empty"
    if is_boothlike(label):
        return "boothlike"
    if RE_FACILITY.search(label):
        return "facility"
    return "text"


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render_pdf(pdf_path: str, dpi: int, out_png: Path, page_index: int = 0,
               max_edge: Optional[int] = None
               ) -> Tuple[np.ndarray, float, float, float]:
    """Rasterise one PDF page to an isolated BGR PNG.

    Returns (bgr, scale, page_w_pt, page_h_pt). `scale` is recomputed from the
    actual rendered width so the PDF-point -> pixel mapping is exact even when
    max_edge clamps the scale down."""
    logger.debug("render_pdf() called pdf_path=%s dpi=%s page_index=%s max_edge=%s",
                 pdf_path, dpi, page_index, max_edge)
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_index]
    w_pt, h_pt = page.get_size()
    scale = dpi / 72.0
    logger.debug("render_pdf: page size w_pt=%s h_pt=%s base scale=%s", w_pt, h_pt, scale)
    if max_edge:
        long_pt = max(w_pt, h_pt)
        if long_pt * scale > max_edge:
            scale = max_edge / long_pt
            logger.debug("render_pdf: clamped scale to %s for max_edge=%s", scale, max_edge)
    logger.info("render_pdf: rendering page at scale=%s", scale)
    bitmap = page.render(scale=scale)
    rgb = np.asarray(bitmap.to_pil().convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), bgr)
    actual_scale = bgr.shape[1] / float(w_pt)
    logger.debug("render_pdf: wrote %s shape=%s actual_scale=%s",
                 out_png, bgr.shape, actual_scale)
    page.close()
    pdf.close()
    return bgr, actual_scale, w_pt, h_pt


def render_raster(image_path: str, out_png: Path,
                  max_edge: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """Read a raster floor plan onto an isolated working copy, downscaling so
    the long edge <= max_edge. Returns (bgr, scale) where scale maps ORIGINAL
    pixels -> working pixels (so detections can be reported in original space
    if ever needed). We keep detections in working space here."""
    logger.debug("render_raster() called image_path=%s max_edge=%s", image_path, max_edge)
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = bgr.shape[:2]
    logger.debug("render_raster: read image w=%s h=%s", w, h)
    scale = 1.0
    if max_edge and max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        logger.debug("render_raster: downscaled to scale=%s new shape=%s", scale, bgr.shape)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), bgr)
    logger.debug("render_raster: wrote %s", out_png)
    return bgr, scale


# --------------------------------------------------------------------------- #
# Text items -- two sources, one shape: {text, bbox_px(x0,y0,x1,y1), center_px}
# --------------------------------------------------------------------------- #
# def extract_text_items_pdf(pdf_path: str, scale: float, page_h_pt: float,
#                            page_index: int = 0) -> List[Dict]:
#     logger.debug("extract_text_items_pdf() called pdf_path=%s scale=%s page_index=%s",
#                  pdf_path, scale, page_index)
#     import pypdfium2 as pdfium
#     pdf = pdfium.PdfDocument(pdf_path)
#     page = pdf[page_index]
#     tp = page.get_textpage()
#     n_rects = tp.count_rects()
#     logger.debug("extract_text_items_pdf: %s text rects in page", n_rects)
#     items: List[Dict] = []
#     for i in range(n_rects):
#         l, b, r, t = tp.get_rect(i)
#         txt = tp.get_text_bounded(left=l, bottom=b, right=r, top=t).strip()
#         if not txt:
#             continue
#         x0, x1 = l * scale, r * scale
#         y0, y1 = (page_h_pt - t) * scale, (page_h_pt - b) * scale   # flip y
#         items.append({"text": txt, "bbox_px": (x0, y0, x1, y1),
#                       "center_px": ((x0 + x1) / 2.0, (y0 + y1) / 2.0)})
#     tp.close()
#     page.close()
#     pdf.close()
#     logger.info("extract_text_items_pdf: %s non-empty text items from %s rects",
#                 len(items), n_rects)
#     return items

def extract_text_items_pdf(pdf_path: str, scale: float, page_h_pt: float,
                           page_index: int = 0) -> List[Dict]:
    logger.debug("extract_text_items_pdf() called pdf_path=%s scale=%s page_index=%s",
                 pdf_path, scale, page_index)
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_index]
    # pdfium renders the CROPBOX region (pixel 0,0 = box top-left), but text
    # rects come back in absolute page space. On a CROPPED PDF the cropbox
    # origin is non-zero, so labels must be shifted by it or every label lands
    # ~origin pixels off its booth and runs into the neighbour. For an
    # un-cropped page (origin 0,0, top == page_h_pt) this is a no-op.
    try:
        bx0, by0, bx1, by1 = page.get_cropbox()
    except Exception:
        bx0, by0, bx1, by1 = 0.0, 0.0, 0.0, page_h_pt
    box_top = by1 if by1 else page_h_pt
    tp = page.get_textpage()
    n_rects = tp.count_rects()
    logger.debug("extract_text_items_pdf: %s text rects in page", n_rects)
    items: List[Dict] = []
    for i in range(n_rects):
        l, b, r, t = tp.get_rect(i)
        txt = tp.get_text_bounded(left=l, bottom=b, right=r, top=t).strip()
        if not txt:
            continue
        x0, x1 = (l - bx0) * scale, (r - bx0) * scale
        y0, y1 = (box_top - t) * scale, (box_top - b) * scale   # flip y
        items.append({"text": txt, "bbox_px": (x0, y0, x1, y1),
                      "center_px": ((x0 + x1) / 2.0, (y0 + y1) / 2.0)})
    tp.close()
    page.close()
    pdf.close()
    logger.info("extract_text_items_pdf: %s non-empty text items from %s rects",
                len(items), n_rects)
    return items


_EASYOCR_READER = None


def _get_easyocr():
    logger.debug("_get_easyocr() called")
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        logger.info("_get_easyocr: initializing EasyOCR reader (en, gpu=False)")
        import easyocr
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _EASYOCR_READER


def extract_text_items_ocr(bgr: np.ndarray, ocr_max_edge: int = 2600) -> List[Dict]:
    """One EasyOCR pass over the whole working image -> text_items in WORKING
    pixel coords. We downscale for the OCR call (speed) and scale the returned
    boxes back up, so the items align with the detection image."""
    logger.debug("extract_text_items_ocr() called shape=%s ocr_max_edge=%s",
                 bgr.shape, ocr_max_edge)
    h, w = bgr.shape[:2]
    s = 1.0
    img = bgr
    if max(w, h) > ocr_max_edge:
        s = ocr_max_edge / float(max(w, h))
        img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        logger.debug("extract_text_items_ocr: downscaled for OCR s=%s shape=%s", s, img.shape)
    reader = _get_easyocr()
    logger.info("extract_text_items_ocr: running EasyOCR readtext")
    results = reader.readtext(img, detail=1, paragraph=False)
    logger.debug("extract_text_items_ocr: EasyOCR returned %s raw results", len(results))
    items: List[Dict] = []
    for box, txt, conf in results:
        if not txt or not str(txt).strip():
            continue
        xs = [p[0] / s for p in box]
        ys = [p[1] / s for p in box]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        items.append({"text": str(txt).strip(), "bbox_px": (x0, y0, x1, y1),
                      "center_px": ((x0 + x1) / 2.0, (y0 + y1) / 2.0)})
    logger.info("extract_text_items_ocr: %s non-empty text items from %s results",
                len(items), len(results))
    return items


# --------------------------------------------------------------------------- #
# Spatial assignment + tagging  (one path for both sources)
# --------------------------------------------------------------------------- #
def _rect_area(r):
    x0, y0, x1, y1 = r
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _overlap_area(bbox, r):
    x, y, w, h = bbox
    x0, y0, x1, y1 = r
    ix = max(0.0, min(x + w, x1) - max(x, x0))
    iy = max(0.0, min(y + h, y1) - max(y, y0))
    return ix * iy


def label_booths(booths: List[Dict], text_items: List[Dict]) -> None:
    """Attach text to booths and tag each (boothlike|text|facility|empty).

    A rect goes to every booth that contains its centre; an orphan rect (centre
    in no box) attaches to the box it overlaps most if that covers >=30% of the
    rect. Mutates booths in place."""
    logger.debug("label_booths() called n_booths=%s n_text_items=%s",
                 len(booths), len(text_items))
    owners: Dict[int, List[Dict]] = {id(b): [] for b in booths}
    for ti in text_items:
        cx, cy = ti["center_px"]
        containing = [b for b in booths
                      if b["bbox"][0] <= cx <= b["bbox"][0] + b["bbox"][2]
                      and b["bbox"][1] <= cy <= b["bbox"][1] + b["bbox"][3]]
        if containing:
            for b in containing:
                owners[id(b)].append(ti)
        else:
            ra = _rect_area(ti["bbox_px"]) or 1.0
            best, best_ov = None, 0.0
            for b in booths:
                ov = _overlap_area(b["bbox"], ti["bbox_px"])
                if ov > best_ov:
                    best, best_ov = b, ov
            if best is not None and best_ov >= 0.30 * ra:
                owners[id(best)].append(ti)

    logger.debug("label_booths: assigned %s text rects across %s booths",
                 sum(len(v) for v in owners.values()), len(booths))
    for bo in booths:
        inside = owners[id(bo)]
        inside.sort(key=lambda ti: (round(ti["center_px"][1] / 8.0),
                                    ti["center_px"][0]))
        label = re.sub(r"\s+", " ", " ".join(ti["text"] for ti in inside)).strip()
        bo["pdf_label"] = label
        bo["n_text"] = len(inside)
        bo["text_status"] = tag_from_label(label)
    logger.info("label_booths: tagged %s booths", len(booths))


# --------------------------------------------------------------------------- #
# False-positive policy
# --------------------------------------------------------------------------- #
# Status-based policies: which text_status values survive.
POLICIES = {
    "none":     {"boothlike", "text", "facility", "empty"},
    "facility": {"boothlike", "text", "empty"},
    "textless": {"boothlike", "text"},
    "strict":   {"boothlike"},
}

# Source-based policy for plans where text is unreliable (rasters whose tiny
# labels don't OCR): a real booth is a COLOUR, GEOMETRIC or BORDERED region. The
# bordered pass is what finds WHITE-on-white booths (which the user explicitly
# wants -- "white booth not detected"), so it must survive shape; its blank-cell
# junk is filtered upstream by the density-scaled bordered_min_area_frac floor,
# not here. Any box that DID get a booth-like label is always kept regardless of
# source.
# "bigregion" is here because the big-region merge already made the keep-vs-drop
# decision: only standalone features (region_role="big") reach the booth list --
# hall containers were dropped at merge time -- so a surviving big box is one we
# decided to keep regardless of whether its title (MAIN STAGE, BAR) is boothlike.
_SHAPE_SOURCES = {"color", "opencv_strict", "bordered", "bigregion"}

# Policies selectable on the CLI (adaptive resolves to strict|shape at runtime).
POLICY_SHAPE_SOURCES = {"color", "bordered", "bigregion"}
POLICY_CHOICES = list(POLICIES) + ["shape", "adaptive"]


def apply_policy(booths: List[Dict], policy: str, fill: str = "white") -> Tuple[List[Dict], List[Dict]]:
    logger.debug("apply_policy() called n_booths=%s policy=%s fill=%s",
                 len(booths), policy, fill)
    if policy == "shape":
        logger.debug("apply_policy: shape policy branch")
        def _keep(b):
            src = str(b.get("source", ""))
            # Color and bordered passes don't need text (the pure geometric shapes are reliable)
            if src in POLICY_SHAPE_SOURCES:
                return True
            # OpenCV pass yields a lot of structural noise. Match the old pipeline:
            # it must contain ANY text to survive.
            if src == "opencv_strict" and b.get("text_status") != "empty":
                return True
            return b.get("text_status") == "boothlike"
        kept = [b for b in booths if _keep(b)]
        dropped = [b for b in booths if not _keep(b)]
        logger.info("apply_policy: shape kept=%s dropped=%s", len(kept), len(dropped))
        return (kept, dropped)
    keep = POLICIES[policy]
    logger.debug("apply_policy: status policy keep=%s", keep)
    kept = [b for b in booths if b["text_status"] in keep]
    dropped = [b for b in booths if b["text_status"] not in keep]
    logger.info("apply_policy: %s kept=%s dropped=%s", policy, len(kept), len(dropped))
    return (kept, dropped)


def resolve_adaptive(booths: List[Dict]) -> str:
    """Pick strict vs shape from how well labeling worked. If a healthy share of
    boxes carry a booth-like label, text is trustworthy -> strict. Otherwise the
    labels didn't survive (low-res raster, illegible stall numbers) and gating on
    them would delete real booths -> fall back to shape (colour/geometry)."""
    logger.debug("resolve_adaptive() called n_booths=%s", len(booths))
    n = len(booths) or 1
    n_boothlike = sum(1 for b in booths if b.get("text_status") == "boothlike")
    decision = "strict" if n_boothlike >= max(10, 0.15 * n) else "shape"
    logger.info("resolve_adaptive: n_boothlike=%s threshold=%s -> %s",
                n_boothlike, max(10, 0.15 * n), decision)
    return decision


# --------------------------------------------------------------------------- #
# Visualization (no name labels by default -- boxes must stay legible)
# --------------------------------------------------------------------------- #
STATUS_BGR = {
    "boothlike": (0, 180, 0),
    "text":      (255, 120, 0),
    "facility":  (0, 0, 255),
    "empty":     (150, 150, 150),
}


def _poly(bo: Dict) -> np.ndarray:
    c = bo.get("coordinates")
    if c:
        return np.array(c, dtype=np.int32)
    x, y, w, h = bo["bbox"]
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)


def draw_boxes(bgr: np.ndarray, booths: List[Dict], with_labels: bool = False) -> np.ndarray:
    logger.debug("draw_boxes() called n_booths=%s with_labels=%s shape=%s",
                 len(booths), with_labels, bgr.shape)
    img = bgr.copy()
    thick = max(2, int(round(img.shape[1] / 1400)))
    for bo in booths:
        col = STATUS_BGR.get(bo.get("text_status", "empty"), (150, 150, 150))
        cv2.polylines(img, [_poly(bo)], isClosed=True, color=col, thickness=thick)
    if with_labels:
        logger.debug("draw_boxes: drawing text labels (with_labels=True)")
        fs = max(0.35, img.shape[1] / 5000.0)
        for bo in booths:
            lbl = bo.get("pdf_label", "")
            if not lbl:
                continue
            x, y, w, h = bo["bbox"]
            short = lbl if len(lbl) <= 22 else lbl[:21] + "…"
            cv2.putText(img, short, (int(x) + 3, int(y) + int(16 * fs) + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                        max(1, thick - 1), cv2.LINE_AA)
    y0 = 30
    for st, col in STATUS_BGR.items():
        n = sum(1 for b in booths if b.get("text_status") == st)
        cv2.rectangle(img, (20, y0 - 16), (50, y0 + 4), col, -1)
        cv2.putText(img, f"{st}: {n}", (60, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        y0 += 34
    logger.debug("draw_boxes: drew %s booths with status legend", len(booths))
    return img


def draw_redmap(bgr: np.ndarray, booths: List[Dict], alpha: float = 0.45) -> np.ndarray:
    logger.debug("draw_redmap() called n_booths=%s alpha=%s shape=%s",
                 len(booths), alpha, bgr.shape)
    img = bgr.copy()
    fill = img.copy()
    polys = [_poly(b) for b in booths]
    if polys:
        cv2.fillPoly(fill, polys, (0, 0, 255))
        cv2.addWeighted(fill, alpha, img, 1.0 - alpha, 0, img)
        thick = max(2, int(round(img.shape[1] / 1400)))
        cv2.polylines(img, polys, isClosed=True, color=(0, 0, 255), thickness=thick)
    cv2.putText(img, f"{len(booths)} booths", (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    logger.debug("draw_redmap: rendered %s booth polygons", len(polys))
    return img

"""
Stage 1 -- turn an InputProfile into a concrete DetectionConfig.

This is the "adaptive" brain: from the three profile axes it assembles the
render settings, which ensemble passes run with what knobs, and how booths get
labeled. Each decision carries a short rationale string so a run is explainable.

Design notes (hard-won, see project memory):
  * Dense vector plans do NOT want tiling + very-high DPI. That makes the
    bordered pass fire on faint background grid lines and explodes the count
    (2373 grid-noise boxes on Bharat vs 162 clean at 150 dpi full-image). The
    containment-aware NMS in the production detector already prevents the
    big-box-swallows-everything collapse, so full-image at a modest DPI is both
    cleaner AND faster here. We bump DPI only enough to resolve tiny cells.
  * The final answer is text-labeled booths only (fp_policy=strict), so blank
    white "available" cells that the bordered pass over-detects are dropped at
    the policy stage anyway -- we don't need to suppress them at detection time.
  * demerge_with_bordered is on whenever the colour pass runs, because the colour
    mask's closing fuses abutting same-fill booths into one block; the bordered
    pass re-splits them and demerge lets the split win.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from input_profile import InputProfile

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    # --- render ---
    dpi: int                              # used for PDFs (scale = dpi/72)
    max_edge: int                         # cap rendered/native long edge (px)
    # --- ensemble passes ---
    use_geometric: bool
    use_color: bool
    use_bordered: bool
    bordered_min_area_frac: float
    max_box_area_frac: float
    demerge_with_bordered: bool
    neutral_gray: Optional[Tuple[int, int]]
    close_ksize: int
    iou_threshold: float
    # --- dense tiling ---
    use_tiling: bool                      # crop into overlapping tiles + per-tile detect
    tile: int                             # tile size (px)
    overlap: int                          # tile overlap (px); must exceed largest booth
    # --- big-region pass (companion to tiling) ---
    use_big_pass: bool                    # full-image pass to recover halls/stages tiling drops
    big_pass_max_edge: int                # downscale long edge for the big pass (memory)
    big_min_side_px: int                  # min side (full px) for a box to count as "big"
    big_min_area_frac: float              # min area frac for the big-pass detectors
    big_max_area_frac: float              # max area frac (drop the whole-floor outline)
    big_coverage_thresh: float            # crops fill >= this of a big box -> it's a container
    big_max_inner_labels: int             # distinct label cells over this -> container
    # --- labeling / policy ---
    label_source: str                     # 'vector' | 'ocr' | 'none'
    fp_policy: str                        # final keep policy
    # --- human-readable ---
    preset: str = ""
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        logger.debug("to_dict() called")
        return asdict(self)


def build_config(prof: InputProfile,
                 fp_policy: Optional[str] = None) -> DetectionConfig:
    """fp_policy=None auto-picks 'adaptive' for EVERY source: keep strict
    (text-labeled booths only) WHEN labeling actually worked, fall back to
    colour/geometry shapes when it didn't. Vector PDFs usually have a rich text
    layer so adaptive resolves to strict for them -- but some vector plans render
    booth numbers as paths, leaving a dead text layer (EDEX: 16 text rects -> 0
    boothlike); there adaptive falls back to shape instead of dropping every
    booth. An explicit value always wins."""
    logger.debug("build_config() called profile=%s density=%s booth_fill=%s "
                 "source=%s fp_policy=%s", prof.name, prof.density,
                 prof.booth_fill, prof.source, fp_policy)
    why: List[str] = []

    # ---- render DPI + edge cap + tiling from DENSITY ----
    # Two regimes:
    #  * DENSE -> TILED detection. A whole-page render makes each stall a
    #    microscopic fraction of the image, so the min-area filters reject it and
    #    abutting cells fuse into row-blocks. Rendering BIG (dpi 250, no edge cap)
    #    and cropping into overlapping tiles makes every booth a healthy fraction
    #    of its tile, so the geometric pass traces each cell -- this is what
    #    recovers a full dense grid (OOAK: 140 single-pass -> ~2000 tiled).
    #  * NORMAL/SPARSE -> single full-image pass at a modest DPI (the old, proven
    #    path). Bumping DPI here only adds grid-noise + cost with no recall gain.
    use_tiling = False
    tile = 1800
    overlap = 400
    if prof.density == "dense":
        logger.debug("density 'dense' -> dpi 250, no edge cap, tiling ON "
                     "(tile=%s overlap=%s)", tile, overlap)
        dpi = 250
        max_edge = 0          # no cap: tiling needs full resolution
        use_tiling = True
        why.append("dense -> dpi 250, NO edge cap, TILED detection "
                   f"(tile {tile}/overlap {overlap}): each small booth becomes a "
                   "healthy fraction of its tile so the geometric pass traces "
                   "every cell (single-pass fuses dense rows into blocks)")
    elif prof.density == "normal":
        logger.debug("density 'normal' -> dpi 150, edge 4500, single-pass")
        dpi = 150
        max_edge = 4500
        why.append("normal -> dpi 150, edge 4500, single-pass (reference clean config)")
    else:  # sparse
        logger.debug("density 'sparse' -> dpi 150, edge 4000, single-pass")
        dpi = 150
        max_edge = 4000
        why.append("sparse -> dpi 150, edge 4000, single-pass (few large booths)")

    # ---- big-region pass: companion to TILING ----
    # Tiling structurally drops any box wider/taller than the overlap -- a hall,
    # stage or standalone big booth (BAR, CHRISTMAS MARKET, MAIN STAGE, W52) is
    # seam-clipped in every tile and discarded, so the dense grid comes through
    # but the big features around it vanish. A full-image DOWNSCALED colour+
    # bordered pass recovers them; then coverage arbitration decides each one:
    # if the small crops already fill >= big_coverage_thresh of a big box (or it
    # wraps many distinct labels) it is a hall CONTAINER -> drop it, keep the
    # crops; otherwise the crops barely fill it -> a STANDALONE feature -> keep
    # the big box (and absorb the few stray crops inside). Only meaningful after
    # tiling, so it is gated on use_tiling.
    use_big_pass = use_tiling
    big_pass_max_edge = 10000
    big_min_side_px = 700
    big_min_area_frac = 8e-4
    big_max_area_frac = 0.5
    big_coverage_thresh = 0.15
    big_max_inner_labels = 6
    logger.debug("big-region pass=%s (gated on tiling) coverage_thresh=%s "
                 "max_inner_labels=%s", use_big_pass, big_coverage_thresh,
                 big_max_inner_labels)
    if use_big_pass:
        why.append("dense -> big-region pass ON (full-image downscaled colour+bordered) "
                   f"+ coverage arbitration: crops filling >= {big_coverage_thresh} of a "
                   "big box (or >6 distinct labels) -> it's a hall container, drop it & "
                   "keep crops; else keep the big box as a standalone feature. "
                   "(0.15 not 0.20: real halls measure 0.18-0.50, so 0.20 would keep a "
                   "0.18 hall as one giant box swallowing its grid)")

    # ---- passes + colour knobs from BOOTH_FILL ----
    use_geometric = use_color = use_bordered = True
    demerge = (prof.density == "dense")
    neutral_gray: Optional[Tuple[int, int]] = None
    max_box_area_frac = 0.06
    iou_threshold = 0.4

    # Bordered floor scales with DENSITY (= booth size). The old fixed 0.005 was
    # tuned to drop blank-white-cell junk, but on a dense grid the REAL booths are
    # ~0.05-0.1% of the page -- well under 0.5% -- so 0.005 silently ate them
    # (OOAK: 45 -> 163 booths once the floor stops eating the dense white cells).
    # Grid-noise (the old fear) stays bounded because edge is capped at 4500;
    # high DPI was what exploded it, not a low floor.
    if prof.density == "dense":
        bordered_min_area_frac = 0.0005
    elif prof.density == "normal":
        bordered_min_area_frac = 0.0015
    else:  # sparse -- few large booths, keep the floor high to reject specks
        bordered_min_area_frac = 0.003
    logger.debug("bordered_min_area_frac=%s (scaled to density %s)",
                 bordered_min_area_frac, prof.density)
    why.append(f"{prof.density} -> bordered min-area {bordered_min_area_frac} "
               "(scaled to booth size so small white cells survive)")

    if prof.booth_fill == "colored":
        logger.debug("booth_fill 'colored' -> colour pass primary, demerge=%s",
                     demerge)
        why.append("colored -> colour pass primary; demerge splits fused "
                   "same-fill rows")
    elif prof.booth_fill == "grey":
        logger.debug("booth_fill 'grey' -> open neutral-grey band (140,245)")
        neutral_gray = (140, 245)
        why.append("grey -> open neutral-grey band (140,245) so the colour pass "
                   "captures grey fills it would otherwise skip")
    else:  # white / outline-only
        logger.debug("booth_fill 'white' -> bordered pass load-bearing")
        why.append("white -> bordered pass is load-bearing (geo floods white "
                   "cells as background, colour ignores them)")

    # Mixed-grey plans: a plan can classify 'colored' (its dominant fill) yet
    # still carry a big band of grey booths the colour pass would skip. If grey
    # pixels are a meaningful share, open the neutral-grey band too. tsfloorplan
    # is the motivating case: classified colored, grey_frac 0.0786, grey booths
    # invisible until the band opens.
    if neutral_gray is None and prof.grey_frac >= 0.05:
        logger.debug("mixed-grey: grey_frac=%s >= 0.05 on non-grey class -> "
                     "open neutral-grey band too", prof.grey_frac)
        neutral_gray = (140, 245)
        why.append(f"grey_frac {prof.grey_frac} >= 0.05 on a non-grey class -> "
                   "open neutral-grey band too (mixed grey booths)")

    close_ksize = 3 if prof.density == "dense" else 9
    logger.debug("close_ksize=%s neutral_gray=%s", close_ksize, neutral_gray)

    # ---- labeling from SOURCE; default policy is adaptive for ALL sources ----
    # Adaptive resolves to strict when labeling yields booth-like labels and to
    # shape when it doesn't. Vector PDFs normally resolve to strict (rich text
    # layer); but a vector plan whose booth numbers are vector PATHS, not text
    # (EDEX: 16 text rects -> 0 boothlike), would lose EVERY booth under a hard
    # strict default -- adaptive catches that and falls back to shape.
    if prof.source == "pdf_vector":
        logger.debug("source 'pdf_vector' -> label_source 'vector', "
                     "auto_policy 'adaptive'")
        label_source = "vector"
        why.append("pdf_vector -> labels from the PDF text layer (no OCR)")
        auto_policy = "adaptive"   # strict when the text layer is rich, else shape
    else:
        logger.debug("source '%s' -> label_source 'ocr', auto_policy 'shape'",
                     prof.source)
        label_source = "ocr"
        why.append(f"{prof.source} -> no vector text; EasyOCR labels booth crops")
        auto_policy = "shape"
        why.append("raster -> policy 'shape' (keep colour/geometry booths; raster "
                   "OCR is too sparse to gate on -- recall beats text-precision)")

    if fp_policy is None:
        logger.debug("fp_policy auto-resolved to '%s'", auto_policy)
        fp_policy = 'none'
        why.append(f"policy auto -> '{fp_policy}' "
                   "(strict if labeling yields booth labels, else shape)")
    else:
        logger.debug("fp_policy forced to '%s'", fp_policy)
        why.append(f"policy forced -> '{fp_policy}'")

    cfg = DetectionConfig(
        dpi=dpi, max_edge=max_edge,
        use_geometric=use_geometric, use_color=use_color, use_bordered=use_bordered,
        bordered_min_area_frac=bordered_min_area_frac,
        max_box_area_frac=max_box_area_frac,
        demerge_with_bordered=demerge,
        neutral_gray=neutral_gray,
        close_ksize=close_ksize,
        iou_threshold=iou_threshold,
        use_tiling=use_tiling,
        tile=tile,
        overlap=overlap,
        use_big_pass=use_big_pass,
        big_pass_max_edge=big_pass_max_edge,
        big_min_side_px=big_min_side_px,
        big_min_area_frac=big_min_area_frac,
        big_max_area_frac=big_max_area_frac,
        big_coverage_thresh=big_coverage_thresh,
        big_max_inner_labels=big_max_inner_labels,
        label_source=label_source,
        fp_policy=fp_policy,
        preset=prof.name,
        rationale=why,
    )
    logger.info("build_config() done preset=%s dpi=%s tiling=%s big_pass=%s "
                "label_source=%s fp_policy=%s (%d rationale lines)", cfg.preset,
                cfg.dpi, cfg.use_tiling, cfg.use_big_pass, cfg.label_source,
                cfg.fp_policy, len(why))
    return cfg


if __name__ == "__main__":
    import sys
    from input_profile import characterize
    for p in sys.argv[1:]:
        prof = characterize(p)
        cfg = build_config(prof)
        print(f"\n{p}\n  profile: {prof.name}\n  preset : {cfg.preset}")
        for r in cfg.rationale:
            print(f"    - {r}")

#!/usr/bin/env python3
"""
Adaptive booth pipeline -- orchestrator.

    input file --> [profile] --> [config] --> [render] --> [detect]
                --> [label] --> [tag] --> [policy] --> outputs

The pipeline LOOKS at the input and configures itself. A vector PDF of a dense
white grid and a raster photo of a colour-coded hall take different paths
automatically -- no flags required. (A VLLM could later replace `characterize`
to pick the same config; everything downstream is unchanged.)

Reuses the production EnsembleDetector (via _detectors) so detection stays the
battle-tested code; this file owns only the adaptive wiring + I/O.

Outputs under <outdir>/<stem>/:
    render_<dpi>dpi.png / render_native.png   detector input (isolated copy)
    <stem>_booths.json     profile, config, every tagged booth, kept/dropped
    <stem>_final.png       kept (strict) booths, status-coloured, NO name labels
    <stem>_labeled.png     kept booths WITH labels (sanity)
    <stem>_redmap.png      kept booths as a red overlay (fast eyeball)
    <stem>_all.png         ALL detected booths tagged (debug, policy=none view)
    <stem>_textmap.png     raw text rects (vector PDFs only)

Security: only reads the input + the imported detector package; never touches
the production .env / S3. Works on an isolated rendered copy of the source.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2

# adaptive layer
from input_profile import characterize, InputProfile
from config import build_config, DetectionConfig
import labeling as L
import tiling as TL
# known-good detectors (single source of truth)
from _detectors import (EnsembleDetector, OpenCVDetector, ColorDetector,
                        BorderedCellDetector)


def _build_detector(cfg: DetectionConfig) -> EnsembleDetector:
    """All sub-detectors run with run_ocr=False: detection is pure geometry and
    labeling happens explicitly afterward (vector layer / EasyOCR). Avoids a
    tesseract dependency in the hot path and is faster. The bordered pass
    self-gates on the presence of white cells, so it is harmless on fully
    colour-coded plans even when enabled."""
    return EnsembleDetector(
        use_geometric=cfg.use_geometric,
        use_color=cfg.use_color,
        use_bordered=cfg.use_bordered,
        bordered_min_area_frac=cfg.bordered_min_area_frac,
        max_box_area_frac=cfg.max_box_area_frac,
        demerge_with_bordered=cfg.demerge_with_bordered,
        iou_threshold=cfg.iou_threshold,
        geometric=OpenCVDetector(run_ocr=False),
        color=ColorDetector(run_ocr=False, neutral_gray=cfg.neutral_gray, close_ksize=cfg.close_ksize),
        bordered=BorderedCellDetector(run_ocr=False),
    )


def _build_tiled_detector(cfg: DetectionConfig) -> EnsembleDetector:
    """Per-tile detector for the DENSE path. Built fresh per call so detect_tiled
    can reuse one instance across tiles. The min-area fractions are of the TILE
    (not the page), so the bordered floor is dropped to 0 and the bordered cell
    floor is the validated 1.5e-4 of tile area -- small dense stalls survive.
    max_box_area_frac still prunes a hall-outline traced as one giant box per
    tile before it can swallow nested booths."""
    return EnsembleDetector(
        use_geometric=cfg.use_geometric,
        use_color=cfg.use_color,
        use_bordered=cfg.use_bordered,
        bordered_min_area_frac=0.0,
        max_box_area_frac=cfg.max_box_area_frac,
        color_neutral_gray=cfg.neutral_gray,
        iou_threshold=cfg.iou_threshold,
        geometric=OpenCVDetector(run_ocr=False),
        color=ColorDetector(run_ocr=False, neutral_gray=cfg.neutral_gray, close_ksize=cfg.close_ksize),
        bordered=BorderedCellDetector(run_ocr=False, min_area_frac=1.5e-4),
    )


def run(input_path: str, outdir: str, page_index: int = 0,
        fp_policy: Optional[str] = None,
        prof: Optional[InputProfile] = None,
        verbose: bool = True) -> Dict:
    stem = Path(input_path).stem
    out = Path(outdir) / stem
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    # ---- 0. profile (skip if caller already did it) ----
    t0 = time.perf_counter()
    if prof is None:
        prof = characterize(input_path)
    log(f"[profile] {prof.name}  "
        f"(col={prof.colored_frac} grey={prof.grey_frac} nGeo={prof.n_geo} "
        f"medAF={prof.median_area_frac}) in {prof.elapsed_s}s")
    if prof.notes:
        for n in prof.notes:
            log(f"[profile]   note: {n}")

    # ---- 1. config ----
    cfg = build_config(prof, fp_policy=fp_policy)
    log(f"[config]  preset='{cfg.preset}' dpi={cfg.dpi} edge={cfg.max_edge} "
        f"passes=(geo={cfg.use_geometric},col={cfg.use_color},bor={cfg.use_bordered}) "
        f"bmaf={cfg.bordered_min_area_frac} mbaf={cfg.max_box_area_frac} "
        f"demerge={cfg.demerge_with_bordered} grey={cfg.neutral_gray} "
        f"label={cfg.label_source}")
    for r in cfg.rationale:
        log(f"[config]   - {r}")

    # ---- 2. render (isolated working copy) ----
    t = time.perf_counter()
    if prof.is_pdf:
        render_png = out / f"render_{cfg.dpi}dpi.png"
        bgr, scale, w_pt, h_pt = L.render_pdf(input_path, cfg.dpi, render_png,
                                              page_index, cfg.max_edge)
        log(f"[render]  {bgr.shape[1]}x{bgr.shape[0]}px @ {cfg.dpi}dpi "
            f"(page {w_pt:.0f}x{h_pt:.0f}pt, scale {scale:.3f}) "
            f"in {time.perf_counter()-t:.1f}s")
    else:
        render_png = out / "render_native.png"
        bgr, scale = L.render_raster(input_path, render_png, cfg.max_edge)
        w_pt = h_pt = None
        log(f"[render]  {bgr.shape[1]}x{bgr.shape[0]}px (raster, scale {scale:.3f}) "
            f"in {time.perf_counter()-t:.1f}s")

    # ---- 3. detect ----
    t = time.perf_counter()
    if cfg.use_tiling:
        booths = TL.detect_tiled(bgr, out, lambda: _build_tiled_detector(cfg),
                                 tile=cfg.tile, overlap=cfg.overlap,
                                 iou_threshold=cfg.iou_threshold, log=log)
    else:
        det = _build_detector(cfg)
        booths = det.detect(str(render_png))
    log(f"[detect]  {len(booths)} booths in {time.perf_counter()-t:.1f}s")

    # ---- 3b. text items (extracted now: the big-region merge counts how many
    #          distinct labels sit inside each big candidate, and labeling reuses
    #          them) ----
    if cfg.label_source == "vector":
        text_items = L.extract_text_items_pdf(input_path, scale, h_pt, page_index)
    elif cfg.label_source == "ocr":
        text_items = L.extract_text_items_ocr(bgr)
    else:
        text_items = []

    # ---- 3c. big-region pass: tiling drops any box wider/taller than the
    #          overlap (seam-clipped in every tile), so halls/stages/standalone
    #          big booths vanish. Recover them full-image, then keep crops where
    #          a big box merely wraps a dense grid (coverage >= thresh) and keep
    #          the big box only where the crops barely fill it. ----
    if cfg.use_big_pass:
        t = time.perf_counter()
        big = TL.detect_big_regions(bgr, out, cfg.neutral_gray,
                                    cfg.big_pass_max_edge, cfg.big_min_area_frac,
                                    cfg.big_max_area_frac, cfg.big_min_side_px,
                                    log=log)
        booths = TL.merge_big_regions(booths, big, text_items,
                                      coverage_thresh=cfg.big_coverage_thresh,
                                      max_inner_labels=cfg.big_max_inner_labels,
                                      label_cell_px=max(120, bgr.shape[1] // 80),
                                      log=log)
        n_big = sum(1 for b in booths if b.get("region_role") == "big")
        log(f"[big]     {n_big} standalone big booths kept, {len(booths)} total "
            f"in {time.perf_counter()-t:.1f}s")

    by_src: Dict[str, int] = {}
    for b in booths:
        by_src[b.get("source", "?")] = by_src.get(b.get("source", "?"), 0) + 1
    log(f"[detect]  sources {by_src}")

    # ---- 4. label + tag ----
    t = time.perf_counter()
    if cfg.label_source in ("vector", "ocr"):
        L.label_booths(booths, text_items)
        src = "vector" if cfg.label_source == "vector" else "EasyOCR"
        log(f"[label]   {len(text_items)} {src} text rects "
            f"in {time.perf_counter()-t:.1f}s")
    else:
        for b in booths:
            b["pdf_label"] = ""
            b["n_text"] = 0
            b["text_status"] = "empty"

    status_counts = {s: sum(1 for b in booths if b.get("text_status") == s)
                     for s in L.STATUS_BGR}
    log(f"[tag]     {status_counts}")
    log("[policy]  would keep:")
    for name in list(L.POLICIES) + ["shape"]:
        k, d = L.apply_policy(booths, name, fill=prof.booth_fill)
        log(f"            {name:9s} -> {len(k):5d} kept  ({len(d)} dropped)")

    # ---- 5. policy (resolve adaptive) + ids ----
    eff_policy = cfg.fp_policy
    if eff_policy == "adaptive":
        eff_policy = L.resolve_adaptive(booths)
        log(f"[policy]  adaptive -> '{eff_policy}' "
            f"(boothlike={status_counts.get('boothlike',0)}/{len(booths)})")
    kept, dropped = L.apply_policy(booths, eff_policy, fill=prof.booth_fill)
    for i, b in enumerate(kept, 1):
        b["id"] = i
    log(f"[final]   policy='{cfg.fp_policy}'"
        + (f" -> '{eff_policy}'" if eff_policy != cfg.fp_policy else "")
        + f": {len(kept)} kept, {len(dropped)} dropped")

    # ---- 6. outputs ----
    cv2.imwrite(str(out / f"{stem}_final.png"), L.draw_boxes(bgr, kept, with_labels=False))
    cv2.imwrite(str(out / f"{stem}_labeled.png"), L.draw_boxes(bgr, kept, with_labels=True))
    cv2.imwrite(str(out / f"{stem}_redmap.png"), L.draw_redmap(bgr, kept))
    cv2.imwrite(str(out / f"{stem}_all.png"), L.draw_boxes(bgr, booths, with_labels=False))
    if cfg.label_source == "vector":
        txt_img = bgr.copy()
        for ti in text_items:
            x0, y0, x1, y1 = (int(v) for v in ti["bbox_px"])
            cv2.rectangle(txt_img, (x0, y0), (x1, y1), (30, 90, 220), 1)
        cv2.imwrite(str(out / f"{stem}_textmap.png"), txt_img)

    payload = {
        "input": input_path,
        "profile": prof.to_dict(),
        "config": cfg.to_dict(),
        "render_px": [bgr.shape[1], bgr.shape[0]],
        "page_pt": [w_pt, h_pt] if w_pt else None,
        "n_text_items": len(text_items),
        "n_detected": len(booths),
        "status_counts": status_counts,
        "fp_policy_requested": cfg.fp_policy,
        "fp_policy_effective": eff_policy,
        "n_kept": len(kept),
        "kept": kept,
        "dropped": dropped,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    json_path = out / f"{stem}_booths.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=list)
    log(f"[write]   {json_path}")
    log(f"[done]    {out}  ({payload['elapsed_s']}s total)")
    return payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="PDF or image floor-plan path.")
    p.add_argument("--outdir", default="/data/out",
                   help="Output directory root.")
    p.add_argument("--page", type=int, default=0, help="0-based PDF page index.")
    p.add_argument("--fp-policy", default=None, choices=L.POLICY_CHOICES,
                   help="Final keep policy. Default: auto (vector->strict, "
                        "raster->adaptive). strict=text-labeled booths only; "
                        "shape=colour/geometry booths; adaptive=strict-or-shape "
                        "by OCR yield; none=keep all.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(args.input, args.outdir, page_index=args.page, fp_policy=args.fp_policy)
    return 0


if __name__ == "__main__":
    sys.exit(main())

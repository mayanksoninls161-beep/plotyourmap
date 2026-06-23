#!/usr/bin/env python3
"""
Run the adaptive pipeline over a folder of floor plans and summarise.

For every PDF/image it runs pipeline.run(), then prints one comparison table:
the auto-chosen profile, detector counts, status breakdown, kept count and time.
The table is what you read to judge whether the adaptive choices were right per
plan TYPE -- the per-file output folders hold the actual viz to eyeball.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import pipeline as P

logger = logging.getLogger(__name__)

DEFAULT_IN = os.getenv("BATCH_IN", "/data/in")
DEFAULT_OUT = os.getenv("BATCH_OUT", "/data/out")
EXTS = (".pdf", ".png", ".jpg", ".jpeg")


def run_batch(indir: str, outdir: str, fp_policy=None,
              only: List[str] = None) -> List[Dict]:
    logger.debug("run_batch() indir=%s outdir=%s fp_policy=%s only=%s",
                 indir, outdir, fp_policy, only)
    files = sorted(f for f in glob.glob(f"{indir}/*")
                   if Path(f).suffix.lower() in EXTS)
    if only:
        files = [f for f in files if any(o.lower() in Path(f).name.lower() for o in only)]
    logger.info("run_batch: %s files to process", len(files))

    rows: List[Dict] = []
    for f in files:
        name = Path(f).name
        logger.info("run_batch: processing %s", name)
        print("\n" + "=" * 78)
        print(f"### {name}")
        print("=" * 78)
        t0 = time.perf_counter()
        try:
            pay = P.run(f, outdir, fp_policy=fp_policy, verbose=True)
            sc = pay["status_counts"]
            rows.append({
                "file": name,
                "profile": pay["profile"]["name"],
                "render_px": "x".join(map(str, pay["render_px"])),
                "n_detected": pay["n_detected"],
                "boothlike": sc.get("boothlike", 0),
                "text": sc.get("text", 0),
                "facility": sc.get("facility", 0),
                "empty": sc.get("empty", 0),
                "policy": pay.get("fp_policy_effective", "?"),
                "n_kept": pay["n_kept"],
                "elapsed_s": pay["elapsed_s"],
                "ok": True,
            })
        except Exception as e:
            logger.exception("run_batch: pipeline failed for %s", name)
            import traceback
            traceback.print_exc()
            rows.append({"file": name, "profile": "ERROR",
                         "error": f"{type(e).__name__}: {e}",
                         "elapsed_s": round(time.perf_counter() - t0, 1),
                         "ok": False})
    return rows


def print_table(rows: List[Dict]) -> None:
    logger.debug("print_table() rows=%s", len(rows))
    print("\n\n" + "#" * 100)
    print("# ADAPTIVE PIPELINE -- BATCH SUMMARY")
    print("#" * 100)
    hdr = (f"{'file':38s} {'profile':24s} {'render':>11s} {'det':>5s} "
           f"{'booth':>5s} {'text':>5s} {'facil':>5s} {'empty':>6s} "
           f"{'policy':>8s} {'kept':>5s} {'s':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if not r.get("ok"):
            print(f"{r['file'][:38]:38s} {'ERROR':24s} {r.get('error','')[:60]}")
            continue
        print(f"{r['file'][:38]:38s} {r['profile']:24s} {r['render_px']:>11s} "
              f"{r['n_detected']:5d} {r['boothlike']:5d} {r['text']:5d} "
              f"{r['facility']:5d} {r['empty']:6d} {r['policy']:>8s} "
              f"{r['n_kept']:5d} {r['elapsed_s']:6.1f}")


def main(argv=None) -> int:
    logger.debug("main() argv=%s", argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default=DEFAULT_IN)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--fp-policy", default=None, choices=P.L.POLICY_CHOICES,
                    help="Override keep policy; default auto per source.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only files whose name contains any of these substrings.")
    args = ap.parse_args(argv)

    rows = run_batch(args.indir, args.outdir, args.fp_policy, args.only)
    print_table(rows)

    summ = Path(args.outdir) / "batch_summary.json"
    summ.parent.mkdir(parents=True, exist_ok=True)
    with summ.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    logger.info("main: wrote batch summary -> %s (%s rows)", summ, len(rows))
    print(f"\n[batch] summary -> {summ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

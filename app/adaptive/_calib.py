"""Throwaway: print the profiler's verdict + raw stats for every test file."""
import glob
import logging
import os
import sys
from pathlib import Path

from input_profile import characterize

logger = logging.getLogger(__name__)

TEST_DIR = os.getenv("CALIB_DIR", "/data/in")


def main():
    logger.debug("main() called test_dir=%s", TEST_DIR)
    files = sorted(glob.glob(f"{TEST_DIR}/*"))
    logger.info("main: found %s entries in %s", len(files), TEST_DIR)
    hdr = f"{'file':44s} {'profile':26s} {'col':>6s} {'grey':>6s} {'wht':>6s} {'nGeo':>5s} {'medAF':>9s} {'np':>3s} {'s':>5s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for f in files:
        if Path(f).suffix.lower() not in (".pdf", ".png", ".jpg", ".jpeg"):
            continue
        logger.debug("main: characterizing %s", f)
        try:
            p = characterize(f)
            print(f"{Path(f).name[:44]:44s} {p.name:26s} {p.colored_frac:6.3f} "
                  f"{p.grey_frac:6.3f} {p.white_frac:6.3f} {p.n_geo:5d} "
                  f"{p.median_area_frac:9.6f} {p.n_pages:3d} {p.elapsed_s:5.1f}",
                  flush=True)
        except Exception as e:
            logger.exception("main: characterize failed for %s", f)
            print(f"{Path(f).name[:44]:44s} ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

"""
Locate and import the known-good ensemble detector ONCE, for the whole package.

The detection code (OpenCV / color / bordered passes + the containment-aware
NMS) is BAKED into this deployment as a sibling `pipeline/` package, so the
folder is FULLY SELF-CONTAINED -- no host paths and no external checkout. This
module puts that baked package on sys.path and re-exports EnsembleDetector + its
sub-detectors from it, so the adaptive layer always runs the same detectors.

Only the ADAPTIVE layer (profiling, preset selection, labeling/tagging/policy,
orchestration) lives in this folder.
"""
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Roots that contain a `detectors/` package and `utils/geometry.py`. First match
# wins. Every candidate is INSIDE the deploy folder / image -- nothing external:
#   1. $BOOTH_DETECTOR_ROOT     -> explicit override (image sets /app/pipeline)
#   2. <this dir>/../pipeline   -> baked sibling package (works locally + image)
#   3. /app/pipeline            -> baked location inside the image
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    os.getenv("BOOTH_DETECTOR_ROOT", ""),
    str(_HERE.parent / "pipeline"),
    "/app/pipeline",
]


def _resolve_detector_root() -> str:
    logger.debug("_resolve_detector_root() called candidates=%s", _CANDIDATES)
    for c in _CANDIDATES:
        if not c:
            continue
        p = Path(c)
        if (p / "detectors" / "ensemble_detector.py").is_file() and \
           (p / "utils" / "geometry.py").is_file():
            logger.info("_resolve_detector_root: resolved DETECTOR_ROOT=%s", p)
            return str(p)
    raise ImportError(
        "Could not find a detector source with detectors/ensemble_detector.py "
        "and utils/geometry.py. Checked: "
        + ", ".join(c for c in _CANDIDATES if c))


DETECTOR_ROOT = _resolve_detector_root()
if DETECTOR_ROOT not in sys.path:
    sys.path.insert(0, DETECTOR_ROOT)

# Re-export so the rest of the package imports from here, not from a hardcoded
# path. `from _detectors import EnsembleDetector, ...`
from detectors.ensemble_detector import EnsembleDetector          # noqa: E402
from detectors.opencv_detector import OpenCVDetector              # noqa: E402
from detectors.color_detector import ColorDetector               # noqa: E402
from detectors.bordered_detector import BorderedCellDetector      # noqa: E402
from utils.geometry import non_max_suppression                   # noqa: E402

__all__ = [
    "DETECTOR_ROOT",
    "EnsembleDetector",
    "OpenCVDetector",
    "ColorDetector",
    "BorderedCellDetector",
    "non_max_suppression",
]

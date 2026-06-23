import logging

from .opencv_detector import OpenCVDetector
from .sam2_detector import SAM2Detector
from .color_detector import ColorDetector
from .ensemble_detector import EnsembleDetector

logger = logging.getLogger(__name__)
logger.debug("detectors package imported")

__all__ = ['OpenCVDetector', 'SAM2Detector', 'ColorDetector', 'EnsembleDetector']

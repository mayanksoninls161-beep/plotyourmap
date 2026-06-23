import logging

logger = logging.getLogger(__name__)


class SAM2Detector:
    def __init__(self):
        """
        Stub for SAM 2 Automatic Mask Generation.
        Will be implemented in Phase 2.
        """
        logger.debug("__init__() called")
        self.model_loaded = False

    def detect(self, image_path):
        """
        Run SAM 2 grid-prompted segmentation.
        """
        logger.debug("detect() called image_path=%s", image_path)
        if not self.model_loaded:
            logger.debug("detect: model not loaded, skipping Part A")
            print("[Warning] SAM 2 model not loaded. Skipping Part A.")
            return []

        # TODO: Implement SAM 2 AutomaticMaskGenerator
        return []

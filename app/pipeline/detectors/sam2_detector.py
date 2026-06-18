class SAM2Detector:
    def __init__(self):
        """
        Stub for SAM 2 Automatic Mask Generation.
        Will be implemented in Phase 2.
        """
        self.model_loaded = False
        
    def detect(self, image_path):
        """
        Run SAM 2 grid-prompted segmentation.
        """
        if not self.model_loaded:
            print("[Warning] SAM 2 model not loaded. Skipping Part A.")
            return []
            
        # TODO: Implement SAM 2 AutomaticMaskGenerator
        return []

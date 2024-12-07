import cv2
import numpy as np


class Camera:

    def __init__(self, source, width: int = 0, height: int = 0):
        self.camera = source
        self.width = width
        self.cap = None
        self.height = height
        self.setup_capture()

    def setup_capture(self):
        self.cap = cv2.VideoCapture(self.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    # existing methods omitted for brevity

    def get_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame from camera {self.camera}")
        return frame

    def release(self):
        self.cap.release()

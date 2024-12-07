import cv2
import ffmpeg

from shared.logger import log_instance


class RtspStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.stream_thread = None

    def frame_generator(self):
        log_instance().info("Starting frame generator")
        cap = cv2.VideoCapture(self.rtsp_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        cap.release()

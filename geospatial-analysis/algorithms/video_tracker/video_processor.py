import cv2


class VideoProcessor:
    def __init__(self, tracker, detector):
        self.tracker = tracker
        self.detector = detector

    def process_video_stream(self, video_stream):
        pass

    def process_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.tracker.track(frame)
            self.detector.detect(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
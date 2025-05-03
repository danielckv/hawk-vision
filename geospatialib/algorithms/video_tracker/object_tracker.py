class ObjectTracker:
    def __init__(self, tracker_type="kcf"):
        self.tracker_type = tracker_type
        self.tracker = cv2.TrackerKCF_create()

    def init_tracker(self, frame, bbox):
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)

    def update(self, frame):
        return self.tracker.update(frame)

    def get_bbox(self):
        return self.tracker.get_bbox()
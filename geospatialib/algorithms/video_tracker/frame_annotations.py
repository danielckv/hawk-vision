class FrameAnnotations:
    def __init__(self, frame_number, annotations):
        self.frame_number = frame_number
        self.annotations = annotations

    def __str__(self):
        return f"FrameAnnotations(frame_number={self.frame_number}, annotations={self.annotations})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.frame_number == other.frame_number and self.annotations == other.annotations

    def __ne__(self, other):
        return not self.__eq__(other)
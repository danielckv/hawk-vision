import os

import cv2
import numpy as np

from src.utils import CURRENT_DIR_APP

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

ONLY_CLASSES = ["person", "dog"]


class Detection:
    def __init__(self, threshold=0.8):

        models_dir = os.path.join(CURRENT_DIR_APP, 'models')

        self.model = cv2.dnn.readNetFromCaffe(models_dir + '/MobileNetSSD_deploy.prototxt.txt',
                                              models_dir + '/MobileNetSSD_deploy.caffemodel')
        self.threshold = threshold

    def cut_frame_to_object(self, frame, detections):
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.threshold:
                if CLASSES[int(detections[0, 0, i, 1])] == "person":
                    box = detections[0, 0, i, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    return frame[startY:endY, startX:endX]
        return None

    def process_frame(self, frame):
        is_object_detected = False
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (740, 420), 127.5)
        self.model.setInput(blob)
        detections = self.model.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.60:  # Confidence threshold
                # Get label text
                if CLASSES[int(detections[0, 0, i, 1])] in ONLY_CLASSES:
                    is_object_detected = True
                    # Get bounding box coordinates and draw
                    box = detections[0, 0, i, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    label_text = CLASSES[int(detections[0, 0, i, 1])]
                    # Add label and confidence score
                    label = "{}: {:.2f}%".format(label_text, confidence * 100)
                    cv2.putText(frame, label, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame, is_object_detected, detections

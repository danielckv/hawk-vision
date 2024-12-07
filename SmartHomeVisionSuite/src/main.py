import signal
import time

import cv2
import requests

from camera import Camera
from detection import Detection
from out_stream import RTSPFrameStreamer
from utils import load_config_yaml

app_state = {
    'camera': None,
    'detector': None,
    'running': True,
    'zeromq': None
}


def signal_handler(sig, last_frame=None):
    app_state['running'] = False
    print(f"Signal {sig} received. Exiting...")


def send_notification():
    requests.get("http://localhost:8123/motion?Living%20Room")
    print("Notification sent!")


def record_motion(last_record_time, output_stream, original_frame):
    current_time = time.time()
    if last_record_time is not None and (current_time - last_record_time) < 60:
        output_stream.write_frame(original_frame)
    else:
        output_stream.close()
        return False

    return True


def start_output_stream():
    return RTSPFrameStreamer("rtsp://localhost:8554/detections")


def write_frame_output(frame, output):
    output.write_frame(frame)


def main():
    config = load_config_yaml()
    debug = config['debug']

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    signal.signal(signal.SIGINT, signal_handler)
    detector = Detection()
    camera = Camera("rtsp://localhost:8554/webcam", 1280, 720)

    # Variables for recording
    recording = False
    out_stream = None
    person_detected_time = 0

    while True:
        ret, original_frame = camera.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            time.sleep(10)
            camera = Camera("rtsp://localhost:8554/webcam", 1280, 720)
            continue

        original_frame = cv2.resize(original_frame, (740, 420))
        current_time = time.time()

        # Process frame
        frame_with_detections, detected_person, detections = detector.process_frame(original_frame)
        if detected_person:
            original_frame = frame_with_detections
            print("Person detected!")

            if not recording:
                out_stream = start_output_stream()
                recording = True
                send_notification()  # Send alert only at the start
                person_detected_time = time.time()  # Mark the time of detection

            # Apply CLAHE (only if recording)
            yuv_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2YUV)
            yuv_frame[:, :, 0] = clahe.apply(yuv_frame[:, :, 0])
            original_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)

        if out_stream is not None:
            write_frame_output(original_frame, out_stream)

            # Check if 5 minutes have passed since the last person detection
            if current_time - person_detected_time >= 310:
                recording = False
                if out_stream is not None:
                    out_stream.close()
                    out_stream = None

        # Display the resulting frame
        if debug:
            cv2.imshow('frame', frame_with_detections)
        if cv2.waitKey(1) & 0xFF == ord('q') or not app_state['running']:
            break

    camera.release()
    cv2.destroyAllWindows()

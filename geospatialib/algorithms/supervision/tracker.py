import gc
import json
import os
import subprocess
import threading
import time
from shutil import move

import cv2
import ffmpeg
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

import shared.json_zlib
import shared.logger
from algorithms.supervision.object_speed import SpeedTrackerAnalysis
from shared.utils import VideoAnalysisTypeInput

COLORS = sv.ColorPalette.default()

rtsp_output = "rtsp://127.0.0.1:8554/stream"
print = shared.log_instance().info
ROOT_DIR = os.path.abspath(os.curdir)


class SuperVideoTracker:
    def __init__(self):
        self.type = None
        self.yolo_local_model = None
        self.obb_analyzer = True
        self.objects_tracker = True
        self.fps = 0
        self.speed_tracker = None
        self.VERBOSE_DEBUG = os.getenv("VERBOSE_DEBUG", False)
        os.environ["YOLO_DEBUG"] = "False"
        self.byte_tracker = None
        self.slicer = None
        self.objects = None
        self.video_frames_length = 0
        self.progress_tqdm = None
        self.original_frame_height = None
        self.original_frame_width = None
        self.model_threshold = 0.40
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available():
            self.device = "mps"
        self.annotator_box_corner = sv.ColorAnnotator()
        self.label_annotator = sv.BoundingBoxAnnotator()
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS.default(),
            trace_length=100,
            thickness=3
        )
        self.frames_manager = None
        self.annotations_boxes = {}
        self.objects_timeline_counter = {}
        self.timeline_annotations = []
        self.threads_count = 7
        self.smoother_video = sv.DetectionsSmoother()
        self.stream_thread = None

        torch.cuda.empty_cache()
        gc.collect()

        print(f"Threads count: {self.threads_count}")
        print(f"Model loaded on {self.device}")

    def start_stream_thread(self):
        ffmpeg_cmd = [
            'ffmpeg',
            '-threads', '4',
            '-y', '-an',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-s', "{}x{}".format(self.original_frame_width, self.original_frame_height),
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-fflags', 'nobuffer',
            '-rtsp_transport', 'tcp',
            '-f', 'rtsp', rtsp_output
        ]
        print(f"Starting rtsp stream with command: {' '.join(ffmpeg_cmd)}")
        self.stream_thread = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            time.sleep(1)
            if self.stream_thread.poll() is None:
                print("Stream thread is ready")
                break

        threading.Thread(target=self.stream_thread.wait).start()

    def update_video_properties(self, video_path, threshold=0.3):
        self.model_threshold = threshold
        cam = cv2.VideoCapture(video_path)
        fps = int(cam.get(cv2.CAP_PROP_FPS))
        self.video_frames_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.slicer = sv.InferenceSlicer(
            slice_wh=(
                int(self.original_frame_width / 2.4),
                int(self.original_frame_height / 2.4)
            ),
            callback=self.callback_slicer,
            thread_workers=self.threads_count
        )
        self.speed_tracker = SpeedTrackerAnalysis(fps)
        self.byte_tracker = sv.ByteTrack(
            frame_rate=fps,
        )
        self.fps = fps
        print(f"Video frames length: {self.video_frames_length}")
        print(f"Video FPS: {self.fps}")
        print(f"Video original frame width: {self.original_frame_width}")
        print(f"Video original frame height: {self.original_frame_height}")

    def callback_slicer(self, frame: np.ndarray) -> sv.Detections:
        yolo_results = self.yolo_local_model(frame, conf=self.model_threshold, device=self.device)[0]
        yolo_detections = sv.Detections.from_ultralytics(yolo_results)
        yolo_detections = yolo_detections[np.isin(yolo_detections.class_id, self.objects)]
        return yolo_detections

    def get_and_calculate_unique_classes(self):
        unique_classes = {}
        for key in self.objects_timeline_counter.keys():
            if unique_classes.get(key.split("#")[0]) is None:
                unique_classes[key.split("#")[0]] = 1
            else:
                unique_classes[key.split("#")[0]] += 1

        return unique_classes

    def original_frame_callback(self, frame: np.ndarray, index: int) -> np.ndarray:
        all_detections = self.slicer(frame)

        current_percentage = (index / (self.video_frames_length - 1)) * 100
        if current_percentage % 10 == 0:
            print(f"Processing frame: {index}/{self.video_frames_length} - {current_percentage:.2f}%")

        all_detections = all_detections.with_nmm()

        if self.objects_tracker is True:
            all_detections = self.byte_tracker.update_with_detections(all_detections)
            all_detections = self.smoother_video.update_with_detections(all_detections)
            frame = self.trace_annotator.annotate(scene=frame.copy(), detections=all_detections)

        annotator_corner = self.annotator_box_corner.annotate(scene=frame, detections=all_detections)

        detection_index = 0
        labels = []
        frame_detections = []
        current_annotations = {}
        timeline_current_annotations_list = []

        for xyxy, mask, confid, class_id, tracker_id, data in all_detections:
            self.speed_tracker.set_object_id(tracker_id)
            speed = self.speed_tracker.analyze_object_speed(xyxy, index)

            label_name = self.yolo_local_model.names[class_id]

            frame_detections.append({
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
                "label": label_name,
                "class": class_id,
                "speed": round(speed, 2),
                "id": tracker_id,
                "confi": round(confid, 2)
            })

            formated_id_key = f"{label_name}#{tracker_id}"
            timeline_current_annotations = {
                "object_id": tracker_id,
                "type": label_name,
            }

            timeline_current_annotations_list.append(timeline_current_annotations)

            if current_annotations.get(formated_id_key) is None:
                current_annotations[formated_id_key] = 1

            if self.objects_timeline_counter.get(formated_id_key) is None:
                self.objects_timeline_counter[formated_id_key] = 1

            confidence = ""
            if self.VERBOSE_DEBUG == 1:
                confidence = f"| {confid:.2f} | Type: {label_name}"

            if speed < 6:
                speed = 0

            labels.append(
                f"ID: #{tracker_id} | Speed: {speed:.2f} mp/h"
                f"{confidence}")

            detection_index += 1

        current_millisecond = self.get_frame_milliseconds(index)
        current_frame_second = self.get_current_frame_seconds(index)

        if self.annotations_boxes.get(current_frame_second) is None:
            self.annotations_boxes[current_frame_second] = {}

        self.annotations_boxes[current_frame_second].update({index: frame_detections})

        self.timeline_annotations.append({
            "milliseconds": current_millisecond,
            "objects": timeline_current_annotations_list
        })

        current_progress = (index / (self.video_frames_length - 1))
        last_ready_frame = self.label_annotator.annotate(scene=annotator_corner,
                                                         detections=all_detections,
                                                         labels=labels)

        self.send_frame_to_stream(index, last_ready_frame)
        if self.progress_tqdm is not None:
            self.progress_tqdm(current_progress, desc=f"Processing frame: {index}")
        return last_ready_frame

    def get_frame_milliseconds(self, frame_index):
        """
        Calculates the milliseconds of a frame based on its index and FPS.

        Args:
            frame_index (int): The index of the frame (starting from 0).
            fps (float): The video's frames per second.

        Returns:
            int: The milliseconds corresponding to the frame.
        """

        if self.fps <= 0:
            raise ValueError("FPS cannot be zero or negative")

        # Milliseconds per frame
        milliseconds_per_frame = 1 / self.fps * 1000

        # Calculate milliseconds based on frame index
        frame_milliseconds = int(frame_index * milliseconds_per_frame)

        return frame_milliseconds

    def get_current_frame_seconds(self, frame_index):
        return frame_index // self.fps

    def send_frame_to_stream(self, index, frame: np.ndarray):
        if self.stream_thread is not None:
            print(f"Writing frame to pipe stream: {index}")
            try:
                self.stream_thread.stdin.write(frame.tobytes())
            except Exception as e:
                print(f"Error writing frame to pipe stream: {e}")
                print("restart stream thread")
                self.stream_thread = None

    def start_local_stream_dispatcher(self, input_stream):
        index = 0
        cap = cv2.VideoCapture(input_stream)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.original_frame_callback(frame, index)
            index += 1
        cap.release()

    def start_file_dispatcher(self, source_path, target_path):
        sv.process_video(source_path=source_path, target_path=target_path, callback=self.original_frame_callback)
        self.byte_tracker = None
        self.slicer = None

        target_path_dir = os.path.dirname(target_path)
        name_of_file = target_path.split("/")[-1].split(".")[0]
        print(f"Saving annotations to {target_path_dir}/{name_of_file}_annotations.json")

        if not os.path.exists(f"{target_path_dir}/output_files"):
            os.makedirs(f"{target_path_dir}/output_files")

        with open(f"{target_path_dir}/{name_of_file}_video_frames_annotations.json", "w") as f:
            json.dump(self.annotations_boxes, f, cls=shared.json_zlib.NumpyEncoder)

        with open(f"{target_path_dir}/output_files/{name_of_file}_timeline_annotations.json", "w") as f:
            json.dump(self.timeline_annotations, f, cls=shared.json_zlib.NumpyEncoder)

        with open(f"{target_path_dir}/output_files/{name_of_file}_all_detections.json", "w") as f:
            json.dump(self.get_and_calculate_unique_classes(), f, cls=shared.json_zlib.NumpyEncoder)

        print("Converting video to mp4")
        full_source_file = f"{target_path_dir}/{name_of_file}_tmp.mp4"

        try:
            # add frame rate
            ffmpeg_command = [
                'ffmpeg',
                '-i', source_path,
                '-vf', f"scale={self.original_frame_width}:{self.original_frame_height}",
                '-aspect', f"{self.original_frame_width / self.original_frame_height}",
                '-c:v', 'libx264',
                '-an',
                '-r', str(self.fps),
                '-pix_fmt', 'yuvj420p',
                '-preset', 'ultrafast',
                full_source_file
            ]
            process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
            print('Conversion successful!')
        except ffmpeg.Error as e:
            print(f'Error during conversion: {e}')

        # create thumbnail from video source_path
        try:
            stream = ffmpeg.input(source_path, ss=1)
            stream = ffmpeg.output(stream, f"{target_path_dir}/thumbnail.jpg", vframes=1)
            ffmpeg.run(stream, overwrite_output=True)  # Overwrite the output file
            print('Thumbnail created successfully!')
        except ffmpeg.Error as e:
            print(f'Error creating thumbnail: {e}')

        if os.path.exists(source_path):
            os.remove(source_path)
        move(full_source_file, source_path)

        return self.annotations_boxes

    def prepare_video_file(self, objects_tracker, source_path, obb_analyzer, threshold=0.2, progress=None,
                           type_mode=VideoAnalysisTypeInput.VIDEO, model_type=None):
        self.objects_tracker = objects_tracker
        self.obb_analyzer = obb_analyzer
        self.annotations_boxes = {}

        self.type = type_mode

        if self.speed_tracker is not None:
            self.speed_tracker._positions = {}
            self.speed_tracker._current_speed = {}

        self.update_video_properties(source_path, threshold=threshold)
        self.start_stream_thread()

        print(f"{ROOT_DIR} is the root dir!")

        if self.obb_analyzer is True:
            self.yolo_local_model = YOLO(model=f"{ROOT_DIR}/models/yolov8m-obb.pt")
        else:
            self.yolo_local_model = YOLO(model=f"{ROOT_DIR}/models/yolov8m-obb.pt")

        if model_type is not None:
            print(f"custom model was selected: {model_type}")
            self.yolo_local_model = YOLO(model=f"./models/{model_type}.pt")

        self.progress_tqdm = progress
        self.objects = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 19]

import os
import threading

from algorithms.supervision.tracker import SuperVideoTracker
from shared import log_instance
from shared.utils import VideoAnalysisTypeInput


class VideoAnalyzer:
    def __init__(self, video_path):
        self.thread = None
        self.video_path = None

        log_instance().info("Initializing video analyzer")

        self.input_type = VideoAnalysisTypeInput.VIDEO
        self.model_instance = None
        self.video_path = video_path

        if video_path.startswith("rtsp://"):
            log_instance().info("INPUT RTSP Stream detected")
            self.input_type = VideoAnalysisTypeInput.RTSP
        else:

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file {video_path} does not exist")

            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video file {video_path} is not a file")

            self.file_name = os.path.basename(video_path).split(".")[0]
            self.path_to_output = os.path.dirname(video_path)

            print("Video path: ", video_path)
            print("File name: ", self.file_name)
            print("Path to output: ", self.path_to_output)
        self._load_algorithm()

    def _load_algorithm(self):
        log_instance().info("Loading algorithm")
        self.model_instance = SuperVideoTracker()

    def analyze(self):
        self.model_instance.prepare_video_file(
            True,
            self.video_path,
            False,
            0.29,
            progress=None,
            type_mode=self.input_type)
        if self.input_type == VideoAnalysisTypeInput.RTSP:
            log_instance().info("Starting RTSP Stream")
            self.model_instance.start_local_stream_dispatcher(self.video_path)
        else:
            log_instance().info("Starting video file analysis")
            analyzed_file_output = self.path_to_output + "/" + self.file_name + "_output.mp4"
            self.model_instance.start_file_dispatcher(self.video_path, analyzed_file_output)
        self.model_instance.stream_thread.stdin.close()
        self.thread.terminate()

    def is_running(self):
        return self.thread.is_alive()

    def start(self):
        if self.thread is None:
            self._new_thread()
            self.thread.start()
        else:
            print("Thread already running")

    def _new_thread(self):
        print("Starting new thread")
        self.thread = threading.Thread(target=self.analyze)

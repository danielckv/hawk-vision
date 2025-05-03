import subprocess
import threading
import time

import numpy as np


class OutputStream:
    def __init__(self, output_url, original_frame_width, original_frame_height, fps):
        self.stream_thread = None
        self.output = output_url
        self.original_frame_width = original_frame_width
        self.original_frame_height = original_frame_height
        self.fps = fps

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
            '-f', 'rtsp', self.output
        ]

        print("Starting stream thread")

        self.stream_thread = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            time.sleep(1)
            if self.stream_thread.poll() is None:
                print("Stream thread is ready")
                break

        print("Stream thread is ready")

        threading.Thread(target=self.stream_thread.wait).start()

    def write(self, frame: np.ndarray):
        if self.stream_thread is not None:
            try:
                self.stream_thread.stdin.write(frame.tobytes())
            except Exception as e:
                print(f"Error writing frame to pipe stream: {e}")
                print("restart stream thread")
                self.stream_thread = None

    def flush(self):
        self.stream_thread.stdin.close()

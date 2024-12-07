from datetime import datetime as dt

from localServer.analyzer.roller import VideoAnalyzer

video_analyzer: VideoAnalyzer


def start_analysis(video_path):
    global video_analyzer
    print(f"Starting analysis on {video_path} at {dt.now()}")
    try:
        video_analyzer = VideoAnalyzer(video_path)
        video_analyzer.start()
    except Exception as e:
        print(f"Error starting analysis: {e}")
        return False
    return True


def start_rtsp_stream():
    global video_analyzer
    rtsp_path = "rtsp://127.0.0.1:8554/stream1"
    print(f"Starting RTSP Stream on {rtsp_path} at {dt.now()}")
    try:
        video_analyzer = VideoAnalyzer(rtsp_path)
        video_analyzer.start()
    except Exception as e:
        print(f"Error starting RTSP Stream: {e}")
        return False
    return True


def check_if_stream_finished():
    if video_analyzer is not None:
        return video_analyzer.is_running()
    return False

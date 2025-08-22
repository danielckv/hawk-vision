import json
import os

from flask import Response

from shared.rpc_client import RPCClient
from videoStreamer.server.base import Base

"""
@api {post} /analysis/video-stream Start Analysis Stream
@apiName StartAnalysisStream
@apiGroup Analysis
@apiVersion 1.0.0
@apiDescription This endpoint starts the analysis of a video stream
@apiBody {String} rtsp_url The rtsp url of the video stream
@apiBody {String} file_path The rtsp url of the video stream
@apiBody {String} model The model to be used for the analysis
@apiSuccess {String} status The status of the request
@apiSuccess {String} message The message of the request
@apiSampleRequest https://localhost:4242/__api__/v1/analysis/video-stream
    {
        "rtsp_url": "rtsp://[username]:[password]@[ip]:[port]/h264_ulaw.sdp",
        "model": "model1"
    }
@apiSuccessExample {Object} Success-Response:
    HTTP/1.1 200 OK
    {
        "status": "success",
        "message": "Analysis started successfully"
    }
"""


class StartAnalysisStreamRequest(Base):
    http_method_type = "post"

    def __init__(self):
        super().__init__()
        self.rpc_video_thread = None
        self.local_rpc = RPCClient('127.0.0.1')
        self.local_rpc.connect()

    def get_video_stream(self):
        return self.request.json.get("rtsp_url")

    def get_model(self):
        return self.request.json.get("model")

    def get_file_path(self):
        return self.request.json.get("file_path")

    def get(self, action):
        super().get(action)
        if action == "start":
            return self.invoke()
        file_path = os.path.join(os.path.dirname(__file__), "../public/index.html")
        html_file_contents = open(file_path, "r")
        return Response(html_file_contents, mimetype="text/html")

    def invoke(self):
        if self.get_video_stream():
            self.local_rpc.start_rtsp_stream()
        else:
            self.local_rpc.start_analysis(self.get_file_path())
        return Response(json.dumps({
            "status": "success",
            "video_url": "http://localhost:8888/stream/index.m3u8"
        }),
            mimetype="application/json"
        )

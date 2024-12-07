from flask import Flask
from flask_restful import Api

from shared.singleton import Singleton


class VideoStreamerServer(metaclass=Singleton):
    def __init__(self, port):
        self.api = None
        self.port = port
        self.app = Flask('rius-video-streamer')
        self.api = Api(app=self.app, prefix='/__api__/v1')

    def boot_server(self):
        print(f"Server running on port {self.port}")
        self.app.run(port=self.port, debug=True, host='0.0.0.0')

    def get_app(self):
        return self.api

import argparse
from videoStreamer.server.stream_api import StartAnalysisStreamRequest
from videoStreamer.server.models_api import TrainedModelsRequest
from videoStreamer.modules import bootstrap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, help='port to run the server on', default='4242')
    options = parser.parse_args()

    server_instance = bootstrap.VideoStreamerServer(options.port)
    server_instance.get_app().add_resource(TrainedModelsRequest, "/models/<action>")
    server_instance.get_app().add_resource(StartAnalysisStreamRequest, "/analysis/video-stream/<action>")
    print("Routes loaded successfully")

    server_instance.boot_server()




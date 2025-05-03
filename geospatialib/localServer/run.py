import argparse

import shared
from localServer.rpc.rpc_server import RPCServer
from localServer.rpc.worker import start_analysis, check_if_stream_finished, start_rtsp_stream

model_instance: None
print = shared.log_instance().info


def bootstrap_server():
    server = RPCServer()
    server.register_method(start_analysis)
    server.register_method(check_if_stream_finished)
    server.register_method(start_rtsp_stream)

    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-type', type=str, help='Type of worker to run', default='rpc')
    options = parser.parse_args()

    if options.worker_type == 'rpc':
        print('Running local rpc worker')
        bootstrap_server()
    elif options.worker_type == 'queue':
        print('Running cli worker')
    else:
        print('Invalid worker type')

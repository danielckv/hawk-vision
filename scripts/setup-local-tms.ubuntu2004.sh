#!/bin/bash

# Check that we are in the data-services root directory before starting the streams
WORKSPACE=$(dirname "$PWD/data-services")
echo "Workspace: $WORKSPACE"

# check if there is python environment directory "venv"
if [ -d "$WORKSPACE/venv" ]; then
    echo "Python environment directory exists"
    source venv/bin/activate
else
    echo "Please make sure that you have created a python environment directory"
    exit 1
fi

# function that spawn the videoStreamer and localServer
startStreams() {
  echo "Starting the videoStreamer and localServer"
  python3.10 videoStreamer/main.py &
  python3.10 localServer/run.py --worker-type=rpc &
  echo "Streams started"
}

if [ -d "$WORKSPACE/videoStreamer" ] && [ -d "$WORKSPACE/localServer" ]; then
    echo "Starting the streams"
    startStreams
else
    echo "Please make sure that you are in the data-services root directory"
    exit 1
fi
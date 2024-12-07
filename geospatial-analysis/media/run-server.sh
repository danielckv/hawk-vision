#!/bin/bash

printf "Starting MediaMTX Server\n"
current_dir=$(pwd)
printf "Current directory: $current_dir\n"

if [ ! -d "mediamtx" ]; then
    cd media || exit
fi

# Check if the current architecture is linux
if [ "$(uname)" == "Linux" ]; then
    echo "Running on Linux"
    # Run the server
    cd mediamtx/mediamtx_v1.5.1_linux_amd64 || exit
    ./mediamtx ../mediamtx.conf.yml
fi

# Check if the current architecture is windows
if [ "$(uname)" == "Windows" ]; then
    echo "Running on Windows"
    # Run the server
    cd mediamtx/mediamtx_v1.5.1_windows_amd64 || exit
    ./mediamtx ../mediamtx.conf.yml
fi

# Check if the current architecture is mac
if [ "$(uname)" == "Darwin" ]; then
    echo "Running on Mac"
    # Run the server
    cd mediamtx/mediamtx_v1.5.1_darwin_arm64 || exit
    ./mediamtx ../mediamtx.conf.yml
fi
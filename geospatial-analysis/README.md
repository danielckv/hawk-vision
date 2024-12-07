## Data Analysis Services - RIUS Platform

![](./algorithms/s2anet/demo/network.png)

This service uses the Flask framework and YOLO (You Only Look Once) for object detection tasks. The service has been tested on Python 3.10.

### Dependencies

Before you start, please make sure you have Python 3.10 installed on your machine. If you don't, follow the instructions [here](https://www.python.org/downloads/).

### Setting Up

1. Clone the repository:
    ```
    git clone <repository_url>
    cd <repository_directory>
    ```

2. It is recommended to set up a virtual environment for the project:
    ```
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Set environment variables:
    ```
    export FLASK_APP=app.py
    export FLASK_ENV=development
    export RABBITMQ_HOST=amqp://[USER:PASSWORD]@localhost:5672
    export REDIS_HOST=localhost
    export REDIS_PORT=6379
    ```

5. Install VisionML dependencies:
    ```
    pip install torchvision>=0.8.1 torch>=1.7.0 tensorflow>=2.13.0
    ```

run the background worker:
```angular2html
python3 localServer/run.py --worker-type=rpc
```

run the media-server:
```angular2html
./assets/mediamtx/run-server.sh
```
2. Open your web browser and navigate to:
    ```
    http://localhost:5000
    ```

### Using the Service

- Once the system is up and running, you can use the service to detect objects in video with RabbitMQ queue:
    ```
    analyze_video
    ```
- The service provides an API endpoint at `/detect`. It is a POST endpoint that accepts an image file and returns object detection results.
- The exact usage depends on the client, but here's an example with curl:
    ```bash
    curl -X POST -F "image=@path_to_your_image.jpg" http://localhost:5000/detect
    ```

## Testing

To run the unit tests:

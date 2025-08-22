import asyncio
import json
import os
import sys

import shared.rabbitmq_client as rabbitmq_client
from services.blobs_manager import upload_file_to_s3
from services.local_queue_processor import enqueue_upload_to_s3
from shared.logger import log_instance

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def upload_sets_to_s3(bucket, blob_path):
    print(f'Uploading blob: {blob_path}')
    upload_file_to_s3(bucket, blob_path, blob_path)


class UploadDatasets(rabbitmq_client.RabbitMQClientInterface):
    def __init__(self):
        log_instance().info("Inference worker created")
        super().__init__("upload_dataset")

    def on_message(self, message):
        log_instance().info("Received message")
        json_body = json.loads(message.body.decode("utf-8"))

        path_files_to_upload = json_body["results_path"]
        for file in path_files_to_upload:
            enqueue_upload_to_s3("video-analysis", file, func=upload_sets_to_s3)

        log_instance().info("Finished uploading datasets")
        message.ack()


async def main():
    rabbitmq_connection = rabbitmq_client.RabbitMQClientConnection()
    await rabbitmq_connection.setup(os.getenv("RABBITMQ_HOST"))
    await rabbitmq_connection.register(UploadDatasets())


if __name__ == "__main__":
    log_instance().info("Starting inference worker")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.run_forever()
    log_instance().info("Inference worker finished")

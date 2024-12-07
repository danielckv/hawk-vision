import asyncio
import json
import os
import sys
import ffmpeg
import aiormq

from services.blobs_manager import download_file_from_s3, upload_file_to_s3

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from shared.json_zlib import NumpyEncoder
from algorithms.supervision.tracker import SuperVideoTracker
import shared.rabbitmq_client as rabbitmq_client
from shared.logger import log_instance


class InferenceWorker(rabbitmq_client.RabbitMQClientInterface):
    def __init__(self):
        super().__init__("AnalyzeVideoRequest")
        log_instance().info("Inference worker created")
        self.video_supervision = SuperVideoTracker()

    async def dispatch(self, message, routing_key: str = ""):
        message = json.dumps(message, cls=NumpyEncoder).encode()
        await self.channel.basic_publish(
            message,
            routing_key=routing_key,
            properties=aiormq.spec.Basic.Properties(
                delivery_mode=1,
            )
        )

    async def finish(self, tag):
        await self.channel.basic_ack(tag)

    def prepare_video_codec_upload(self, bucket, key_file_name, blob_path):
        file_output = f"{blob_path}.mp4"
        (
            ffmpeg
            .input(blob_path)
            .output(file_output, vcodec='libx264', crf=20, preset='veryslow', pix_fmt='yuv420p')
            .run()
        )

        print(f'Uploading blob: {blob_path}')
        upload_file_to_s3(bucket, key_file_name, file_output)

    ###
    # on_message
    # {
    #   "file": "https://imisightvideos.s3.eu-west-1.amazonaws.com/videos/bBW-ZQ8-dnw-7oa-BE.mp4"
    # }
    async def on_message(self, message: aiormq.abc.DeliveredMessage):
        log_instance().info("Received message")

        json_body = json.loads(message.body.decode("utf-8"))
        inner_json = json.loads(json_body["body"])
        file_url = inner_json["file"]
        headers = json_body["headers"]

        file_s3_path = file_url.replace("https://", "").split(".")
        full_s3_path_bucket = file_s3_path[0]
        full_s3_path_file_name = file_url.split("videos/")[1]

        log_instance().info(f"Downloading file from S3 {full_s3_path_bucket} {full_s3_path_file_name}")
        local_path = download_file_from_s3(full_s3_path_bucket, "videos/" + full_s3_path_file_name)

        local_path_file_name = local_path.split("/")[2].split(".")[0]
        desktop_path = os.path.expanduser("~/Desktop")
        output_file = os.path.join(desktop_path, local_path_file_name + ".mp4")
        annotations = self.video_supervision.detect_and_track_video_objects(local_path, output_file)

        json_body_results = {
            "headers": headers,
            "body": {
                "file": f"https://imisightvideos.s3.eu-west-1.amazonaws.com/annotated_videos/{local_path_file_name}_analyzed.mp4",
                "frames": annotations,
            }
        }
        path_for_analyzed_video_file = f"annotated_videos/{local_path_file_name}_analyzed.mp4"
        self.prepare_video_codec_upload("imisightvideos", path_for_analyzed_video_file, output_file)

        log_instance().info(f"Finished inference for video file: {local_path_file_name}")

        await self.dispatch(json_body_results, "AnalyzeVideoResponse")
        await self.finish(message.delivery.delivery_tag)


async def __service_main__():
    rabbitmq_connection = rabbitmq_client.RabbitMQClientConnection()
    await rabbitmq_connection.setup(os.getenv("RABBITMQ_HOST"))
    await rabbitmq_connection.register(InferenceWorker())


if __name__ == "__main__":
    log_instance().info("Starting inference worker")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(__service_main__())
    loop.run_forever()
    log_instance().info("Inference worker finished")

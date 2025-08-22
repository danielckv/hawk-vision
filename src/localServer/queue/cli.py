import argparse
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from metaseg import SegAutoMaskPredictor
from shared.logger import log_instance
from services.blobs_manager import download_file_from_s3


def cli_app(cli_args):
    file_s3_path = cli_args.file_source.replace("https://", "").split(".")
    full_s3_path_bucket = file_s3_path[0]
    full_s3_path_file_name = cli_args.file_source.split("videos/")[1]
    log_instance().info(f"Downloading file from S3 {full_s3_path_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-source', type=str, help='Path to the video file URL.')
    opt = parser.parse_args()

    cli_app(opt)

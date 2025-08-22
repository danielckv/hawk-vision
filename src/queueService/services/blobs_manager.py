import os
import uuid

import boto3

# load  access_key and secret from env variables to aws s3 client

s3 = boto3.client('s3',
                  aws_access_key_id=os.getenv("ACCESS_KEY"),
                  region_name=os.getenv("REGION"),
                  aws_secret_access_key=os.getenv("SECRET_KEY"))


def download_file_from_s3(bucket_name, key):
    local_path = '/tmp/video-raw-' + str(uuid.uuid4()) + '.mp4'
    s3.download_file(bucket_name, key, local_path)
    return local_path


def upload_file_to_s3(bucket_name, key, local_path):
    s3.upload_file(local_path, bucket_name, key, ExtraArgs={
        'ContentType': 'video/mp4',
        'ContentDisposition': 'inline'
    })
    os.remove(local_path)

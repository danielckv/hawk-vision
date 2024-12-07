import os
from redis import Redis
from rq import Queue

REDIS_PORT = os.getenv('REDIS_PORT', 6379)
REDIT_HOST = os.getenv('REDIS_HOST', 'localhost')

q = Queue(connection=Redis().from_url(f'redis://{REDIT_HOST}:{REDIS_PORT}/10'))


def enqueue_video_tiles(tiles_path, func):
    q.enqueue(func, tiles_path)


def enqueue_upload_to_s3(bucket, key_file_name, blob_path, func):
    q.enqueue(func, bucket, key_file_name, blob_path)

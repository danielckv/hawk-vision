import os
import boto3

from dotenv import load_dotenv

load_dotenv('.env')  # load environment variables from .env file

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_BUCKET_REGION = os.getenv('AWS_BUCKET_REGION')

MODEL_BUCKET = os.environ.get('MODEL_BUCKET')
MODEL_LOCATION = os.environ.get('MODEL_LOCATION')
MODEL_TO_PUSH = os.environ.get('MODEL_TO_PUSH')

s3 = boto3.resource('s3',
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_KEY,
                    region_name=AWS_BUCKET_REGION)

bucket = s3.Bucket(MODEL_BUCKET)
file_to_upload = os.path.basename(MODEL_TO_PUSH)
print(file_to_upload)
upload_key = os.path.join(MODEL_LOCATION, file_to_upload)
print(upload_key)
s3.meta.client.upload_file(MODEL_TO_PUSH, MODEL_BUCKET, upload_key)

print("File to uploadddd:" , (file_to_upload))
print(f"Upload keyyyy: {upload_key}")

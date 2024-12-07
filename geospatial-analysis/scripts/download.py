import boto3
import os

# Fetching AWS credentials and S3 bucket details from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_AKID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SK')
AWS_BUCKET_REGION = os.getenv('AWS_BUCKET_REGION')
TRAINING_BUCKET = os.getenv('TRAINING_BUCKET')
DATA_SET = os.getenv('DATA_SET')
LOCAL_DIR = os.getenv('LOCAL_DIR', '/app/datasets/datasets/pointAI')  # Set to your desired local directory

# Print environment variables for debugging
print(f"AWS_ACCESS_KEY_ID: {AWS_ACCESS_KEY_ID}")
print(f"AWS_SECRET_ACCESS_KEY: {AWS_SECRET_ACCESS_KEY}")
print(f"AWS_BUCKET_REGION: {AWS_BUCKET_REGION}")
print(f"TRAINING_BUCKET: {TRAINING_BUCKET}")
print(f"DATA_SET: {DATA_SET}")
print(f"LOCAL_DIR: {LOCAL_DIR}")

# Check if required environment variables are set
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_BUCKET_REGION, TRAINING_BUCKET, DATA_SET]):
    raise ValueError("Environment variables AWS_AKID, AWS_SK, AWS_BUCKET_REGION, TRAINING_BUCKET, and DATA_SET must be set.")

def download_folder_from_s3(bucket_name, s3_folder, local_dir):
    try:
        # Initialize boto3 client with credentials and region
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_BUCKET_REGION
        )

        if not os.path.exists(local_dir):
            print(f"creating local dir: ${local_dir}")
            os.makedirs(local_dir)

        # List files in the S3 folder
        result = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_folder)

        # Check if the folder exists in S3
        if 'Contents' not in result:
            raise FileNotFoundError(f"No files found in S3 bucket '${bucket_name}' with prefix '${s3_folder}'")

        # Download each file
        for content in result.get('Contents', []):
            file_key = content['Key']
            if not file_key.endswith("/"):
                local_file_name = os.path.basename(file_key)
                local_file_path = f"{local_dir}/{local_file_name}"

                # if not os.path.exists(os.path.dirname(local_file_path)):
                #     os.makedirs(os.path.dirname(local_file_path))

                print(f"Downloading {file_key} to {local_file_path}")
                s3.download_file(bucket_name, file_key, local_file_path)
                print(f"Downloaded {file_key} to {local_file_path}")

    except Exception as e:
        print(f"Error downloading files: {e}")
        raise

# Usage example
download_folder_from_s3(TRAINING_BUCKET, DATA_SET, LOCAL_DIR)

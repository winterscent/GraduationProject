import boto3

BUCKET_NAME = "your-bucket-name"
s3 = boto3.client('s3')

def upload_to_s3(file_path):
    try:
        s3.upload_file(file_path, BUCKET_NAME, file_path)
    except Exception as e:
        raise RuntimeError(f"Error uploading to S3: {str(e)}")

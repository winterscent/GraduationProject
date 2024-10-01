import boto3
import os

# S3 설정
BUCKET_NAME = "your-bucket-name"
s3 = boto3.client('s3')

# S3 파일 업로드 함수
def upload_to_s3(file_path):
    try:
        file_name = os.path.basename(file_path)
        s3.upload_file(file_path, BUCKET_NAME, file_name)
        return f"https://{BUCKET_NAME}.s3.amazonaws.com/{file_name}"
    except Exception as e:
        raise RuntimeError(f"Error uploading to S3: {str(e)}")

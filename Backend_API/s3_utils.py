import boto3
import uuid
import os

def upload_to_s3(file_name, bucket, object_name=None):
    # S3 리소스 생성
    s3 = boto3.client('s3')

    # 파일 이름에서 UUID 사용, 형식: UUID_masked.csv
    if object_name is None:
        original_extension = os.path.splitext(file_name)[1]
        object_name = f"conversationanalysis/{uuid.uuid4()}_masked{original_extension}"

    # 업로드
    try:
        s3.upload_file(file_name, bucket, object_name)
        print(f"File {file_name} uploaded successfully to {bucket}/conversationanalysis/{object_name}.")
        # 업로드한 파일의 URL 반환
        return f"https://{bucket}.s3.amazonaws.com/conversationanalysis/{object_name}"
    except Exception as e:
        print(f"Failed to upload {file_name}. Error: {e}")
        return None

import boto3

def upload_to_s3(file_name, bucket, object_name=None):
    # S3 리소스 생성
    s3 = boto3.client('s3')

    # 업로드
    try:
        if object_name is None:
            object_name = file_name
        s3.upload_file(file_name, bucket, object_name)
        print(f"File {file_name} uploaded successfully to {bucket}/{object_name}.")
    except Exception as e:
        print(f"Failed to upload {file_name}. Error: {e}")

import boto3

def upload_to_s3(file_path, bucket='cell2cell-bucket', key=None):
    if key is None:
        key = file_path.split('/')[-1]
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, key)
    print(f"S3 Upload: {key} → s3://{bucket}/{key}")
    return f"s3://{bucket}/{key}"
import json
import boto3
import pandas as pd
from io import StringIO
from src.model.drift_detector import detect_drift
from src.model.train import train_model  # Import your train func
from src.config import S3_BUCKET, S3_RAW_PREFIX, S3_LOGS_PREFIX, PSI_THRESHOLD

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Step 1: Download baseline & current batch from S3
    baseline_obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{S3_RAW_PREFIX}baseline.csv')
    current_obj = s3.get_object(Bucket=S3_BUCKET, Key=f'{S3_RAW_PREFIX}current_batch.csv')
    
    baseline_df = pd.read_csv(StringIO(baseline_obj['Body'].read().decode('utf-8')))
    current_df = pd.read_csv(StringIO(current_obj['Body'].read().decode('utf-8')))
    
    # Step 2: Detect drift
    drifts, has_drift = detect_drift(baseline_df, current_df)
    
    # Step 3: If drift, trigger retrain
    if has_drift:
        print(f"Drift detected! PSI exceeds {PSI_THRESHOLD}")
        # Trigger SageMaker job (use boto3.sagemaker)
        sagemaker = boto3.client('sagemaker')
        sagemaker.create_training_job(
            TrainingJobName='churn-retrain-job',
            AlgorithmSpecification={'TrainingImage': 'your-image', 'TrainingInputMode': 'File'},
            # Add your S3 inputs, output path, etc.
        )
        # After training, deploy (call deploy_model.py)
    else:
        print("No drift - model stable")
    
    # Log results to CloudWatch/S3
    log_df = pd.DataFrame(drifts).T
    csv_buffer = StringIO()
    log_df.to_csv(csv_buffer)
    s3.put_object(Bucket=S3_BUCKET, Key=f'{S3_LOGS_PREFIX}drift_log_{pd.Timestamp.now().strftime("%Y%m%d")}.csv', Body=csv_buffer.getvalue())
    
    return {
        'statusCode': 200,
        'body': json.dumps({'drift_detected': has_drift, 'details': drifts})
    }
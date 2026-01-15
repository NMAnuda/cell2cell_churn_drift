import boto3
import json

def trigger_retrain(psi_avg, bucket='cell2cell-bucket'):
    lambda_svc = boto3.client('lambda')
    payload = json.dumps({'psi_avg': psi_avg, 'bucket': bucket})
    response = lambda_svc.invoke(FunctionName='drift-alert-lambda', Payload=payload)
    print(f"Lambda Triggered: drift-alert-lambda (status: {response['StatusCode']})")
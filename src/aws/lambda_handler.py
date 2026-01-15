import json

def lambda_handler(event, context):
    psi_avg = event.get('psi_avg', 0)
    print(f"Lambda Alert: PSI {psi_avg:.3f} — Drift detected!")
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'Alert sent', 'psi': psi_avg})
    }
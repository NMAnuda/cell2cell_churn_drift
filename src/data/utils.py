import pandas as pd
import json

def load_csv(path):
    return pd.read_csv(path)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
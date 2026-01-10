
import pandas as pd


def generate_batches(df, n_batches=5):

    if len(df) < n_batches:
        raise ValueError(f"DataFrame too small ({len(df)} rows) for {n_batches} batches.")
    
    batch_size = len(df) // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i < n_batches - 1 else len(df)  # Last batch takes remainder
        batch = df.iloc[start:end].reset_index(drop=True)
        batches.append(batch)
    
    print(f"Generated {n_batches} even batches: sizes {[len(b) for b in batches]}")
    return batches  # batch[0] = baseline, batch[1:] = incoming for drift


# Example usage (for testing)
if __name__ == "__main__":
    # Dummy test data
    import numpy as np
    dummy_df = pd.DataFrame({
        'feature1': np.random.randn(25000),
        'target': np.random.choice([0, 1], 25000, p=[0.84, 0.16])
    })
    batches = generate_batches(dummy_df, n_batches=5)
    print("Test successful!")
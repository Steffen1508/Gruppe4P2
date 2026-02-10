import pandas as pd
# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/syvai/pii-dataset-eng/data/train-00000-of-00001.parquet")
print(df.head())
df.info()

import os
import pandas as pd

files = [
    "dataset/instruments.csv",
    "dataset/log_info.csv"
]

for file in files:

    print("\n" + "=" * 70)
    print("FILE:", file)
    print("=" * 70)

    if not os.path.exists(file):
        print("File not found")
        continue

    df = pd.read_csv(file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 10 Rows:")
    print(df.head(10))

from pathlib import Path
import pandas as pd

print("Starting Wine Dataset Preprocessing Pipeline... \n")

DATA_DIR = Path('data')
raw_file = DATA_DIR / "Churn_Modeling.csv"
processed_file = DATA_DIR / 'processed_churn.csv'

# 1. Load Raw Data
print(f"Reading raw data from: {raw_file}...")
df = pd.read_csv(raw_file)
print(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print("data cols:", df.columns)
print()

# 2. Drop Unnecessary Columns
print("Dropped unnecessary columns: RowNumber, CustomerId, Surname")
df = df.drop_duplicates()
df = df.drop(['CustomerId', 'Surname'] , axis = 1)
    

label_counts = df["Exited"].value_counts()
for label, count in label_counts.items():
    pct = (count / len(df)) * 100
    print(f"   • {label:<8}: {count:>4} samples ({pct:.1f}%)")

# # 3. Drop Original Exited Column
# print("Dropping original 'Exited' numeric column...")
# df.drop(columns=['Exited'], inplace=True)

# 4. Save Processed Data
print(f"Saving processed dataset to: {processed_file}...")
processed_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_file, index=False)

print(f"\nPreprocessing completed successfully!")
print(f"   • Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
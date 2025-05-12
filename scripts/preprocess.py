import pandas as pd

csv_filename = "data/skin_disease_dataset.csv"

# Load dataset
df = pd.read_csv(csv_filename)

# Remove missing values
df.dropna(inplace=True)

# Save cleaned CSV
cleaned_csv = "data/cleaned_skin_disease_dataset.csv"
df.to_csv(cleaned_csv, index=False)

print(f"Cleaned CSV saved at {cleaned_csv}")

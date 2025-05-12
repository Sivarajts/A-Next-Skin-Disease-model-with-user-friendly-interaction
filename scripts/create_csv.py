import os
import pandas as pd
import dateutil
print("dateutil is installed")

# Set the path to your dataset
dataset_path = "F:archive"
csv_filename = "data/skin_disease_dataset.csv"

# Create a list for filenames and labels
data = []

for disease_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, disease_folder)

    if os.path.isdir(folder_path):
        for img in os.listdir(folder_path):
            img_path = os.path.join(disease_folder, img)
            data.append([img_path, disease_folder])

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data, columns=["image_path", "label"])
df.to_csv(csv_filename, index=False)

print(f"CSV file saved at {csv_filename}")

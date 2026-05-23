import yaml
import pandas as pd

# Load Params
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

data_path = params["base"]["data_path"]

dialogues = []
with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        parts = line.split(" +++$+++ ")
        if len(parts) >= 5:
            text = parts[4].strip()
            dialogues.append(text)

print(f"Loaded {len(dialogues)} lines of dialogue.")

# Save the ENTIRE dataset to CSV
df = pd.DataFrame({"text": dialogues})
df.to_csv("data/clean_dialogues.csv", index=False)
print("Saved entire dataset to clean_dialogues.csv")
import json
import os
from collections import Counter

LABEL_DIR = os.path.join("DENTEX", "DENTEX", "disease", "label")

counts = Counter()
examples = {}

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".json"):
        continue
    with open(os.path.join(LABEL_DIR, file), encoding="utf-8") as f:
        data = json.load(f)
    for shape in data["shapes"]:
        label = shape["label"]
        parts = label.split("-")
        key = "-".join(parts[:-1]) if len(parts) >= 3 else label  # drop tooth number
        counts[key] += 1
        examples.setdefault(key, label)

print("Unique test label types (without tooth number):")
for key, n in counts.most_common():
    print(f"  {key:30s} count={n:5d}   example: {examples[key]}")
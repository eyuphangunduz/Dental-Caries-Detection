import os
import zipfile

ROOT = "DENTEX"

# 1) Extract all zip files into the folder they are in
for folder, _, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".zip"):
            path = os.path.join(folder, f)
            print("Extracting:", path)
            with zipfile.ZipFile(path) as z:
                z.extractall(folder)

# 2) Show folder structure and file counts
print("\n--- FOLDER STRUCTURE ---")
for folder, subfolders, files in os.walk(ROOT):
    if ".cache" in folder:
        continue
    depth = folder.count(os.sep)
    images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    others = [f for f in files if f not in images]
    print("  " * depth + os.path.basename(folder) + f"/  ({len(images)} images)")
    for f in others:
        print("  " * (depth + 1) + f)
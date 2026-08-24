from pathlib import Path

source = Path("dataset.txt")
target = Path("dataset_small.txt")

max_chars = 50_000_000  # 50MB

with open(source, "r", encoding="utf-8") as f:
    data = f.read(max_chars)

with open(target, "w", encoding="utf-8") as f:
    f.write(data)

print("Created smaller dataset!")
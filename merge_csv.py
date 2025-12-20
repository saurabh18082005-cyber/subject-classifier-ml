import csv

files = [
    "biology_300.csv",
    "physics_300.csv",
    "chemistry_300.csv",
    "math_300.csv",
    "computer_300.csv"
]

rows = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "text": row["text"],
                "label": row["subject"]
            })

with open("training_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(rows)

print("training_data.csv created with", len(rows), "rows")

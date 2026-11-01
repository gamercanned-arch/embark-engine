import json

with open("scratch/lfm3_tensors.json") as f:
    tensors = json.load(f)

print("=== LAYER 0 (ShortConv) ===")
for t in tensors:
    if t["name"].startswith("blk.0."):
        print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, bytes={t['bytes']}")

print("\n=== LAYER 2 (Attention) ===")
for t in tensors:
    if t["name"].startswith("blk.2."):
        print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, bytes={t['bytes']}")

print("\n=== NON-LAYER TENSORS ===")
for t in tensors:
    if not t["name"].startswith("blk."):
        print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, bytes={t['bytes']}")

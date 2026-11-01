import json

with open("scratch/lfm3_mmproj_tensors.json") as f:
    tensors = json.load(f)

print(f"Total mmproj tensors: {len(tensors)}")
for t in tensors:
    print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, bytes={t['bytes']}")

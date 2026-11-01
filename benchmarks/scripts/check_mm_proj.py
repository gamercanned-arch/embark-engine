import json

with open("scratch/mmproj_tensors.json") as f:
    mm_tensors = json.load(f)

for t in mm_tensors:
    name = t["name"]
    if not (name.startswith("v.blk.") or name.startswith("a.blk.")):
        print(f"Non-block tensor: {name} shape={t['shape']} type={t['type']}")

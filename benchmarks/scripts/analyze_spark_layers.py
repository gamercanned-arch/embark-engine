import json

with open("scratch/spark_tensors.json") as f:
    tensors = json.load(f)

print(f"Total tensors: {len(tensors)}")
non_blk = []
layer_types = {}

for t in tensors:
    name = t["name"]
    if not name.startswith("blk."):
        non_blk.append(t)
    else:
        parts = name.split(".")
        layer_idx = int(parts[1])
        subname = ".".join(parts[2:])
        if subname not in layer_types:
            layer_types[subname] = []
        layer_types[subname].append((layer_idx, t["type"], t["shape"], t["bytes"]))

print("\nNon-block tensors:")
for t in non_blk:
    print(f"  {t['name']}: {t['shape']} {t['type']} ({t['bytes']/(1024*1024):.2f} MB)")

print("\nPer-block tensor types:")
for subname, items in layer_types.items():
    print(f"  {subname} (count={len(items)}): e.g. layer {items[0][0]}: {items[0][1]} {items[0][2]} ({items[0][3]/(1024*1024):.2f} MB)")

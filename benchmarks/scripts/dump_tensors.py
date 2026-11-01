import json

with open("scratch/embeddinggemma_tensors.json") as f:
    tensors = json.load(f)

print(f"EmbeddingGemma tensor count: {len(tensors)}")
print("First 25 tensors:")
for t in tensors[:25]:
    print(f"  {t['name']}: {t['shape']} ({t['type']})")

print("...")
print("Last 25 tensors:")
for t in tensors[-25:]:
    print(f"  {t['name']}: {t['shape']} ({t['type']})")

with open("scratch/mmproj_tensors.json") as f:
    mm_tensors = json.load(f)

print(f"\nMMProj tensor count: {len(mm_tensors)}")
print("Sample MMProj tensors:")
for t in mm_tensors[:20]:
    print(f"  {t['name']}: {t['shape']} ({t['type']})")
print("...")
for t in mm_tensors[-20:]:
    print(f"  {t['name']}: {t['shape']} ({t['type']})")

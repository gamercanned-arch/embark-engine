import json

with open("scratch/embeddinggemma_tensors.json") as f:
    tensors = json.load(f)

tensor_map = {t["name"]: t for t in tensors}

print("=== GLOBAL TENSORS ===")
for name in ["token_embd.weight", "output_norm.weight", "output.weight", "per_layer_model_proj.weight", "per_layer_proj_norm.weight"]:
    if name in tensor_map:
        t = tensor_map[name]
        print(f"  {name}: shape={t['shape']} type={t['type']} bytes={t['bytes']/(1024*1024):.2f}MB")

print("\n=== LAYER TENSORS COMPARISON (blk.0 vs blk.5 vs blk.23) ===")
for blk_idx in [0, 1, 5, 6, 11, 12, 23]:
    print(f"\n--- Block {blk_idx} ---")
    prefix = f"blk.{blk_idx}."
    for name, t in tensor_map.items():
        if name.startswith(prefix):
            short_name = name[len(prefix):]
            print(f"  {short_name}: shape={t['shape']} type={t['type']}")

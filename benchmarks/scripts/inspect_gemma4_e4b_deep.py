import sys
from collections import defaultdict
import gguf
import numpy as np
import json

path = r"C:\Users\abhik\Downloads\gemma-4-E4B-it-Q4_K_M.gguf"
reader = gguf.GGUFReader(path)

print("=" * 60)
print("METADATA DEEP DIVE: Gemma 4 E4B")
print("=" * 60)

metadata = {}
for k, field in reader.fields.items():
    if k.startswith("tokenizer."):
        continue
    val = None
    try:
        if field.types and field.types[0] == gguf.GGUFValueType.ARRAY:
            val = [field.parts[d] for d in field.data]
            if len(val) > 0 and isinstance(val[0], np.ndarray):
                val = [x.tolist() if x.size > 1 else x.item() for x in val]
        elif len(field.parts) > 0:
            part = field.parts[-1]
            if isinstance(part, np.ndarray):
                if part.dtype.kind in ('S', 'U'):
                    val = part.tobytes().decode('utf-8', errors='ignore')
                elif part.size == 1:
                    val = part.item()
                else:
                    val = part.tolist()
            else:
                val = part
    except Exception as e:
        val = f"err: {e}"
    
    metadata[k] = val
    if isinstance(val, list) and len(val) > 10:
        print(f"{k}: list len={len(val)}, first 10={val[:10]}")
    else:
        print(f"{k}: {val}")

layer_tensors = defaultdict(dict)
non_layer_tensors = []
total_bytes = 0

for t in reader.tensors:
    name = t.name
    total_bytes += t.n_bytes
    if "blk." in name:
        parts = name.split(".")
        blk_idx = int(parts[1])
        t_key = ".".join(parts[2:])
        layer_tensors[blk_idx][t_key] = {
            "name": name,
            "shape": list(t.shape),
            "type": t.tensor_type.name,
            "elements": t.n_elements,
            "bytes": t.n_bytes
        }
    else:
        non_layer_tensors.append({
            "name": name,
            "shape": list(t.shape),
            "type": t.tensor_type.name,
            "elements": t.n_elements,
            "bytes": t.n_bytes
        })

print(f"\nTotal layers: {len(layer_tensors)}")
print(f"Total model bytes: {total_bytes} ({total_bytes / (1024**3):.3f} GiB)")

print("\nLayer 0 Tensors:")
for k, v in sorted(layer_tensors[0].items()):
    print(f"  {k:<30} shape={v['shape']} type={v['type']} ({v['bytes']/(1024**2):.2f} MB)")

all_keys = set()
for l_idx, d in layer_tensors.items():
    all_keys.update(d.keys())

print(f"\nAll unique tensor keys across all layers ({len(all_keys)}):")
for k in sorted(all_keys):
    present_in = [l_idx for l_idx, d in layer_tensors.items() if k in d]
    missing_in = [l_idx for l_idx, d in layer_tensors.items() if k not in d]
    if missing_in:
        print(f"  {k:<30}: present in {len(present_in)} layers, MISSING in {missing_in}")
    else:
        print(f"  {k:<30}: present in ALL {len(present_in)} layers")

# Save summary
summary = {
    "metadata": {k: str(v) if not isinstance(v, (int, float, list, dict, str, bool)) else v for k, v in metadata.items()},
    "total_layers": len(layer_tensors),
    "total_bytes": total_bytes,
    "non_layer_tensors": non_layer_tensors,
    "layer_0": {k: layer_tensors[0][k] for k in layer_tensors[0]}
}

with open(r"scratch/gemma4_e4b_arch_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved scratch/gemma4_e4b_arch_summary.json successfully.")

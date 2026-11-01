import sys
from collections import defaultdict
import gguf
import numpy as np

path = r"C:\Users\abhik\Downloads\Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
reader = gguf.GGUFReader(path)

print("=" * 60)
print("METADATA DEEP DIVE")
print("=" * 60)

for k, field in reader.fields.items():
    if k.startswith("tokenizer."):
        continue
    # extract value nicely
    val = None
    try:
        if field.types and field.types[0] == gguf.GGUFValueType.ARRAY:
            val = [field.parts[d] for d in field.data]
            # Convert numpy arrays to list/scalar
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
    
    if isinstance(val, list) and len(val) > 10:
        print(f"{k}: list len={len(val)}, first 10={val[:10]}")
    else:
        print(f"{k}: {val}")

print("\n" + "=" * 60)
print("LAYER ARCHITECTURE & KV INVESTIGATION")
print("=" * 60)

layer_tensors = defaultdict(dict)
for t in reader.tensors:
    name = t.name
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

print(f"Total layers: {len(layer_tensors)}")

# Investigate layer 5 specifically
print("\nLayer 5 Tensors:")
for k, v in sorted(layer_tensors[5].items()):
    print(f"  {k:<30} shape={v['shape']} type={v['type']} ({v['bytes']/(1024**2):.2f} MB)")

# Check which layers have which tensors
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

# Check shapes and types across layers for each tensor key
print("\nShape and Type Variations across Layers:")
for k in sorted(all_keys):
    variations = defaultdict(list)
    for l_idx, d in layer_tensors.items():
        if k in d:
            var_key = (tuple(d[k]["shape"]), d[k]["type"])
            variations[var_key].append(l_idx)
    print(f"\nTensor: {k}")
    for (shape, ttype), layers in variations.items():
        if len(layers) == 48:
            print(f"  shape={shape} type={ttype} in ALL 48 layers")
        else:
            print(f"  shape={shape} type={ttype} in {len(layers)} layers: {layers}")

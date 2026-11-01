import sys
import json
from collections import defaultdict
import gguf

path = r"C:\Users\abhik\Downloads\Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
reader = gguf.GGUFReader(path)

print("=" * 60)
print("1. GGUF METADATA")
print("=" * 60)

metadata = {}
for k, field in reader.fields.items():
    if k.startswith("tokenizer."):
        continue
    # Extract value
    parts = field.parts
    data = field.data
    types = field.types
    if len(types) == 1:
        val = field.contents
    elif hasattr(field, "data") and len(field.data) > 10:
        val = f"<Array len={len(field.data)}>"
    else:
        try:
            val = [field.parts[d] for d in field.data] if field.data else field.contents
        except Exception:
            val = str(field)
    metadata[k] = val
    print(f"{k}: {val}")

arch = metadata.get("general.architecture", "unknown")
print(f"\nArchitecture: {arch}")

print("\n" + "=" * 60)
print("2. TENSOR INVENTORY")
print("=" * 60)

type_counts = defaultdict(int)
type_bytes = defaultdict(int)
type_elements = defaultdict(int)

layer_tensors = defaultdict(list)
non_layer_tensors = []

total_bytes = 0
total_elements = 0

for t in reader.tensors:
    name = t.name
    shape = list(t.shape)
    tensor_type = t.tensor_type.name
    n_elems = t.n_elements
    n_bytes = t.n_bytes
    
    total_bytes += n_bytes
    total_elements += n_elems
    
    type_counts[tensor_type] += 1
    type_bytes[tensor_type] += n_bytes
    type_elements[tensor_type] += n_elems
    
    # Check if layer tensor
    if "blk." in name:
        parts = name.split(".")
        try:
            blk_idx = int(parts[1])
            layer_tensors[blk_idx].append({
                "name": name,
                "shape": shape,
                "type": tensor_type,
                "elements": n_elems,
                "bytes": n_bytes
            })
        except ValueError:
            non_layer_tensors.append((name, shape, tensor_type, n_elems, n_bytes))
    else:
        non_layer_tensors.append((name, shape, tensor_type, n_elems, n_bytes))

print(f"Total Tensors: {len(reader.tensors)}")
print(f"Total Elements: {total_elements:,}")
print(f"Total Bytes: {total_bytes:,} ({total_bytes / (1024**3):.3f} GiB)")

print("\nQuantization Distribution:")
for qtype, b in sorted(type_bytes.items(), key=lambda x: x[1], reverse=True):
    cnt = type_counts[qtype]
    pct = (b / total_bytes) * 100
    elems = type_elements[qtype]
    bpw = (b * 8) / elems if elems > 0 else 0
    print(f"  {qtype:<10}: {b / (1024**2):9.2f} MB ({pct:5.2f}%) | {cnt:4d} tensors | {bpw:.2f} bpw")

print("\nNon-Layer Tensors:")
for name, shape, ttype, elems, b in non_layer_tensors:
    print(f"  {name:<45} {str(shape):<20} {ttype:<10} {b / (1024**2):8.2f} MB")

print("\nPer-Layer Breakdown (Layers: {}):".format(len(layer_tensors)))
# Inspect Layer 0 and Layer 1 and last layer to see tensor types
for layer_idx in [0, 1, 2, len(layer_tensors)-1]:
    if layer_idx in layer_tensors:
        print(f"\n--- Layer {layer_idx} ---")
        layer_b = sum(t["bytes"] for t in layer_tensors[layer_idx])
        print(f"Total Layer Size: {layer_b / (1024**2):.2f} MB")
        for t in layer_tensors[layer_idx]:
            print(f"  {t['name']:<45} {str(t['shape']):<20} {t['type']:<10} {t['bytes'] / (1024**2):8.2f} MB")

# Check for mixed precision across layers
layer_quant_profile = defaultdict(lambda: defaultdict(int))
for l_idx, tensors in layer_tensors.items():
    for t in tensors:
        short_name = ".".join(t["name"].split(".")[2:])
        layer_quant_profile[l_idx][short_name] = t["type"]

# Check differences between layers
base_profile = layer_quant_profile[0]
diff_layers = []
for l_idx, prof in layer_quant_profile.items():
    if prof != base_profile:
        diff_layers.append(l_idx)

if diff_layers:
    print(f"\n[ALERT] Mixed-precision / varying architecture found across layers: {diff_layers}")
    for l_idx in diff_layers[:5]:
        print(f"Layer {l_idx} diffs from Layer 0:")
        for k in set(base_profile.keys()).union(set(layer_quant_profile[l_idx].keys())):
            t0 = base_profile.get(k, "MISSING")
            tl = layer_quant_profile[l_idx].get(k, "MISSING")
            if t0 != tl:
                print(f"  {k}: Layer 0 has {t0}, Layer {l_idx} has {tl}")
else:
    print("\nAll layers share the identical quantization and tensor structure!")

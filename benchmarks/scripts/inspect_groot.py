import sys
from collections import defaultdict
import gguf

path = r"C:\Users\abhik\Downloads\GrootTeam_Groot-19B-A3B-261023-IQ3_XXS.gguf"
reader = gguf.GGUFReader(path)

print("=" * 60)
print("1. GGUF METADATA")
print("=" * 60)

for k, field in reader.fields.items():
    if k.startswith("tokenizer."):
        continue
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
    print(f"{k}: {val}")

arch = reader.fields.get("general.architecture")
if arch:
    print(f"\nArchitecture: {arch.contents}")

print("\n" + "=" * 60)
print("2. TENSOR INVENTORY")
print("=" * 60)

type_counts = defaultdict(int)
type_bytes = defaultdict(int)
layer_types = defaultdict(lambda: defaultdict(int))

total_bytes = 0
for t in reader.tensors:
    total_bytes += t.n_bytes
    ttype = t.tensor_type.name
    type_counts[ttype] += 1
    type_bytes[ttype] += t.n_bytes
    if "blk." in t.name:
        parts = t.name.split(".")
        layer_idx = int(parts[1])
        subname = ".".join(parts[2:])
        layer_types[layer_idx][subname] = (t.shape, ttype, t.n_bytes)

print(f"Total Tensors: {len(reader.tensors)}")
print(f"Total Bytes: {total_bytes:,} ({total_bytes / (1024**3):.3f} GiB)")

print("\nQuantization Distribution:")
for qtype, b in sorted(type_bytes.items(), key=lambda x: x[1], reverse=True):
    cnt = type_counts[qtype]
    pct = (b / total_bytes) * 100
    print(f"  {qtype:<12}: {b / (1024**2):9.2f} MB ({pct:5.2f}%) | {cnt:4d} tensors")

print("\nLayer count:", len(layer_types))
print("Examining layer 0 tensors:")
for k, v in layer_types[0].items():
    print(f"  {k:<35} {str(v[0]):<20} {v[1]:<10} {v[2]/(1024**2):.2f} MB")

print("\nExamining layer 1 tensors:")
for k, v in layer_types[1].items():
    print(f"  {k:<35} {str(v[0]):<20} {v[1]:<10} {v[2]/(1024**2):.2f} MB")

# Check which layers have which attention / linear mechanism
print("\nLayer pattern search:")
for l_idx in range(len(layer_types)):
    keys = list(layer_types[l_idx].keys())
    has_attn = any("attn" in k or "self_attn" in k for k in keys)
    has_gdn = any("gdn" in k or "linear_attn" in k or "ssm" in k or "recurrent" in k for k in keys)
    has_moe = any("exps" in k or "expert" in k for k in keys)
    key_summary = [k for k in keys if "norm" not in k]
    print(f"Layer {l_idx:2d}: {len(keys)} tensors, keys: {key_summary[:6]} ...")

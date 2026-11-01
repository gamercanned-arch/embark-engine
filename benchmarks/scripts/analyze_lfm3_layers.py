import json

with open("scratch/lfm3_tensors.json") as f:
    tensors = json.load(f)

layers = {}
other = []
for t in tensors:
    name = t["name"]
    if name.startswith("blk."):
        parts = name.split(".")
        idx = int(parts[1])
        if idx not in layers:
            layers[idx] = []
        layers[idx].append((name, t["shape"], t["type"], t["bytes"]))
    else:
        other.append((name, t["shape"], t["type"], t["bytes"]))

print(f"Total layers: {len(layers)}")
print(f"Other tensors: {len(other)}")
for name, shape, t_type, b in other:
    print(f"  {name}: {shape} {t_type} ({b/1024/1024:.2f} MB)")

shortconv_layers = []
attn_layers = []
for idx in sorted(layers.keys()):
    tensor_names = [x[0] for x in layers[idx]]
    has_conv = any("shortconv" in name for name in tensor_names)
    has_attn = any("attn_q" in name or "attn_output" in name for name in tensor_names)
    layer_bytes = sum(x[3] for x in layers[idx])
    if has_conv:
        shortconv_layers.append(idx)
        print(f"Layer {idx:2d}: SHORTCONV ({len(layers[idx])} tensors, {layer_bytes/1024/1024:.2f} MB) - {[x[0].split('.')[-2]+'.'+x[0].split('.')[-1] for x in layers[idx]]}")
    elif has_attn:
        attn_layers.append(idx)
        print(f"Layer {idx:2d}: ATTENTION ({len(layers[idx])} tensors, {layer_bytes/1024/1024:.2f} MB) - {[x[0].split('.')[-2]+'.'+x[0].split('.')[-1] for x in layers[idx]]}")
    else:
        print(f"Layer {idx:2d}: UNKNOWN ({len(layers[idx])} tensors, {layer_bytes/1024/1024:.2f} MB)")

print(f"\nTotal ShortConv layers: {len(shortconv_layers)}: {shortconv_layers}")
print(f"Total Attention layers: {len(attn_layers)}: {attn_layers}")

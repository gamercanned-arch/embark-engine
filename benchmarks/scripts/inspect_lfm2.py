import gguf

reader = gguf.GGUFReader(r"C:\Users\abhik\Downloads\LFM2.5-350M-Q8_0.gguf")
layer_types = {}
for t in reader.tensors:
    parts = t.name.split('.')
    if parts[0] == 'blk':
        b_idx = int(parts[1])
        sub = '.'.join(parts[2:])
        if b_idx not in layer_types:
            layer_types[b_idx] = []
        layer_types[b_idx].append((sub, t.shape, t.tensor_type))
    else:
        print(f"Global: {t.name:35s} shape={str(t.shape):20s} type={t.tensor_type}")

print(f"\nTotal blocks: {len(layer_types)}")
for b in sorted(layer_types.keys()):
    names = [x[0] for x in layer_types[b]]
    print(f"Block {b:2d}: {names}")

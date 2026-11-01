import gguf

path = r"C:\Users\abhik\Downloads\Ornith-1.5-9B-Q4_K_M.gguf"
r = gguf.GGUFReader(path)
arch = r.fields.get('general.architecture')
arch_str = bytes(arch.data).decode('utf-8') if arch else 'unknown'
print(f"Architecture: {arch_str}")

for k, v in r.fields.items():
    if any(x in k.lower() for x in ['layer', 'head', 'dim', 'context', 'window', 'state', 'ssm', 'gdn', 'delta', 'rnn', 'conv']):
        data = v.data
        if hasattr(data, 'tobytes'):
            try:
                data = bytes(data).decode('utf-8')
            except:
                data = list(data)[:5]
        elif isinstance(data, list) and len(data) > 5:
            data = data[:5]
        print(f"{k}: {data}")

print("\n--- Tensor Names Sample ---")
for t in r.tensors[:25]:
    print(t.name, t.shape, t.tensor_type)

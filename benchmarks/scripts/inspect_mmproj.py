import gguf

path = r"C:\Users\abhik\Downloads\mmproj-F16.gguf"
reader = gguf.GGUFReader(path)

print("=== MMPROJ GGUF METADATA ===")
for k, field in reader.fields.items():
    print(f"  {k} = {field.parts[-1] if len(field.parts) > 0 else 'none'}")

print(f"\nTotal mmproj tensors: {len(reader.tensors)}")
total_bytes = 0
for t in reader.tensors[:15]:
    total_bytes += t.n_bytes
    print(f"  {t.name}: shape={list(t.shape)}, type={t.tensor_type.name}, {t.n_bytes/(1024*1024):.2f} MB")

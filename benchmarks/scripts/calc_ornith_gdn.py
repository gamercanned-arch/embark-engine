import gguf
import sys

path = r"C:\Users\abhik\Downloads\Ornith-1.5-9B-Q4_K_M.gguf"
r = gguf.GGUFReader(path)

print("=== Ornith 1.5 9B Architecture Fields ===", flush=True)
for k, v in r.fields.items():
    if any(x in k.lower() for x in ['block_count', 'ssm', 'gdn', 'layer', 'context_length', 'embedding_length']):
        val = v.data
        if hasattr(val, 'tolist'):
            val = val.tolist()
        print(f"  {k} = {val}", flush=True)

# Also check conv1d and ssm state tensors to get exact state shapes
print("\n=== SSM / GDN Tensors ===", flush=True)
ssm_tensors = [t for t in r.tensors if 'ssm' in t.name]
for t in ssm_tensors[:15]:
    print(f"  {t.name}: shape={t.shape}, type={t.tensor_type}", flush=True)

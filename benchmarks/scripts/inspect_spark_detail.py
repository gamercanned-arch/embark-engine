import gguf
import json

path = r"C:\Users\abhik\Downloads\Spark-X2.6-4B-Q4_K_M.gguf"
r = gguf.GGUFReader(path)

metadata = {}
for k, v in r.fields.items():
    if not k.startswith("tokenizer"):
        val = v.data
        if hasattr(val, 'tolist'):
            val = val.tolist()
        elif hasattr(val, 'tobytes'):
            try:
                val = bytes(val).decode('utf-8')
            except:
                val = str(val)
        metadata[k] = str(val)

tensors_info = []
total_bytes = 0
q4_k_bytes = 0
q4_k_count = 0
type_counts = {}
type_bytes = {}

for t in r.tensors:
    size = t.n_bytes
    total_bytes += size
    t_type = str(t.tensor_type)
    type_counts[t_type] = type_counts.get(t_type, 0) + 1
    type_bytes[t_type] = type_bytes.get(t_type, 0) + size
    if "Q4_K" in t_type:
        q4_k_bytes += size
        q4_k_count += 1
    tensors_info.append({
        "name": t.name,
        "shape": [int(s) for s in t.shape],
        "type": t_type,
        "bytes": size
    })

print(f"Total tensors: {len(tensors_info)}, Total size: {total_bytes / (1024*1024):.2f} MB ({total_bytes / (1024*1024*1024):.2f} GiB)")
print(f"Q4_K tensors: {q4_k_count}, Q4_K size: {q4_k_bytes / (1024*1024):.2f} MB")
print("\n=== Quant Type Breakdown ===")
for t_type, count in type_counts.items():
    print(f"  {t_type}: {count} tensors, {type_bytes[t_type] / (1024*1024):.2f} MB")

print("\n=== Sample Metadata ===")
for k, v in metadata.items():
    print(f"  {k} = {v}")

print("\n=== Layer 0 tensors ===")
for t in tensors_info:
    if t["name"].startswith("blk.0."):
        print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, {t['bytes']/(1024*1024):.2f} MB")

with open(r"scratch/spark_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

with open(r"scratch/spark_tensors.json", "w") as f:
    json.dump(tensors_info, f, indent=2)

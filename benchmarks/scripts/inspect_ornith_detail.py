import gguf
import json

path = r"C:\Users\abhik\Downloads\Ornith-1.5-9B-Q4_K_M.gguf"
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
for t in r.tensors:
    size = t.n_bytes
    total_bytes += size
    t_type = str(t.tensor_type)
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

print("\n=== Sample Metadata ===")
for k in ["general.architecture", "qwen35.block_count", "qwen35.context_length", "qwen35.embedding_length", 
          "qwen35.feed_forward_length", "qwen35.attention.head_count", "qwen35.attention.head_count_kv",
          "qwen35.ssm.conv_kernel", "qwen35.ssm.state_size", "qwen35.ssm.group_count", "qwen35.ssm.time_step_rank",
          "qwen35.ssm.inner_size", "qwen35.full_attention_interval"]:
    if k in metadata:
        print(f"  {k} = {metadata[k]}")

print("\n=== Layer 0 (GDN) tensors ===")
for t in tensors_info:
    if t["name"].startswith("blk.0."):
        print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, {t['bytes']/(1024*1024):.2f} MB")

print("\n=== Layer 3 (Full Attention) tensors ===")
for t in tensors_info:
    if t["name"].startswith("blk.3."):
        print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, {t['bytes']/(1024*1024):.2f} MB")

with open(r"scratch/ornith_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

with open(r"scratch/ornith_tensors.json", "w") as f:
    json.dump(tensors_info, f, indent=2)

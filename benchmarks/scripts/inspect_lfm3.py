import gguf
import json
import os

def inspect(path, name):
    print(f"\n=======================================================")
    print(f"=== INSPECTING {name}: {path} ===")
    print(f"=======================================================")
    r = gguf.GGUFReader(path)
    metadata = {}
    for k, field in r.fields.items():
        if not k.startswith("tokenizer"):
            # Decode field value
            val = field.parts[field.data[0]] if len(field.data) > 0 else None
            # Or use field.data
            try:
                if len(field.data) == 1:
                    raw = field.data[0]
                    if hasattr(raw, 'tolist'):
                        val = raw.tolist()
                    elif isinstance(raw, (bytes, bytearray)):
                        val = bytes(raw).decode('utf-8', errors='ignore')
                    else:
                        val = raw
                else:
                    items = []
                    for idx in field.data:
                        if hasattr(idx, 'tolist'):
                            items.append(idx.tolist())
                        elif isinstance(idx, (bytes, bytearray)):
                            items.append(bytes(idx).decode('utf-8', errors='ignore'))
                        else:
                            items.append(idx)
                    val = items
            except Exception as e:
                val = str(field.data)
            metadata[k] = val
            print(f"  {k} = {val}")

    tensors_info = []
    total_bytes = 0
    type_counts = {}
    type_bytes = {}

    for t in r.tensors:
        size = t.n_bytes
        total_bytes += size
        t_type = str(t.tensor_type)
        type_counts[t_type] = type_counts.get(t_type, 0) + 1
        type_bytes[t_type] = type_bytes.get(t_type, 0) + size
        tensors_info.append({
            "name": t.name,
            "shape": [int(s) for s in t.shape],
            "type": t_type,
            "bytes": size
        })

    print(f"\nTotal tensors: {len(tensors_info)}, Total size: {total_bytes / (1024*1024):.2f} MB ({total_bytes / (1024*1024*1024):.2f} GiB)")
    print("\n=== Quant Type Breakdown ===")
    for t_type, count in type_counts.items():
        print(f"  {t_type}: {count} tensors, {type_bytes[t_type] / (1024*1024):.2f} MB")

    print("\n=== Tensor names sample (first 20) ===")
    for t in tensors_info[:20]:
        print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, {t['bytes']/(1024*1024):.2f} MB")

    print("\n=== Tensor names sample (last 20) ===")
    for t in tensors_info[-20:]:
        print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, {t['bytes']/(1024*1024):.2f} MB")

    with open(f"scratch/{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(f"scratch/{name}_tensors.json", "w") as f:
        json.dump(tensors_info, f, indent=2)

inspect(r"C:\Users\abhik\Downloads\LFM3-VL-2B-Q4_K_M.gguf", "lfm3")
inspect(r"C:\Users\abhik\Downloads\mmproj-LFM3-VL-2B-F16.gguf", "lfm3_mmproj")

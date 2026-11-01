import gguf
import json

def inspect(path, name):
    print(f"\n=======================================================")
    print(f"=== INSPECTING {name}: {path} ===")
    print(f"=======================================================")
    r = gguf.GGUFReader(path)
    metadata = {}
    for k, field in r.fields.items():
        if not k.startswith("tokenizer"):
            parts = []
            for d in field.data:
                part = field.parts[d]
                if hasattr(part, 'tobytes'):
                    part = bytes(part).decode('utf-8', errors='ignore')
                elif hasattr(part, 'tolist'):
                    part = part.tolist()
                parts.append(part)
            val = parts[0] if len(parts) == 1 else parts
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

    print(f"\nTotal tensors: {len(tensors_info)}, Total size: {total_bytes / (1024*1024):.2f} MB")
    for tt, c in type_counts.items():
        print(f"  {tt}: {c} tensors, {type_bytes[tt]/(1024*1024):.2f} MB")

    with open(f"scratch/{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(f"scratch/{name}_tensors.json", "w") as f:
        json.dump(tensors_info, f, indent=2)

if __name__ == "__main__":
    inspect(r"C:\Users\abhik\Downloads\embeddinggemma-2-UD-Q6_K_XL.gguf", "embeddinggemma")
    inspect(r"C:\Users\abhik\Downloads\mmproj-F16 (1).gguf", "mmproj")

import gguf

r = gguf.GGUFReader(r"C:\Users\abhik\Downloads\LFM3-VL-2B-Q4_K_M.gguf")
for k, field in r.fields.items():
    if not k.startswith("tokenizer"):
        val = None
        if len(field.types) == 1:
            part = field.parts[field.data[0]]
            val = part.tolist() if hasattr(part, 'tolist') else bytes(part).decode('utf-8', errors='ignore') if isinstance(part, (bytes, memoryview)) else part
        else:
            items = []
            for idx in field.data:
                p = field.parts[idx]
                items.append(p.tolist() if hasattr(p, 'tolist') else bytes(p).decode('utf-8', errors='ignore') if isinstance(p, (bytes, memoryview)) else p)
            val = items
        print(f"{k} = {val}")

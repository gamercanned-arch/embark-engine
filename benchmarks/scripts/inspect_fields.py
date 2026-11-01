import gguf

r = gguf.GGUFReader(r"C:\Users\abhik\Downloads\embeddinggemma-2-UD-Q6_K_XL.gguf")
for k, f in r.fields.items():
    if not k.startswith("tokenizer"):
        val = []
        for d in f.data:
            part = f.parts[d]
            val.append(part.tolist() if hasattr(part, 'tolist') else part)
        # also print type
        print(f"{k}: type={f.types}, val={val}")

r_mm = gguf.GGUFReader(r"C:\Users\abhik\Downloads\mmproj-F16 (1).gguf")
for k, f in r_mm.fields.items():
    if not k.startswith("tokenizer"):
        val = []
        for d in f.data:
            part = f.parts[d]
            val.append(part.tolist() if hasattr(part, 'tolist') else part)
        print(f"MM: {k}: type={f.types}, val={val}")

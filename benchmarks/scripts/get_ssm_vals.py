import gguf

r = gguf.GGUFReader(r"C:\Users\abhik\Downloads\Ornith-1.5-9B-Q4_K_M.gguf")
for k, f in r.fields.items():
    if 'ssm' in k or 'block_count' in k or 'context' in k:
        try:
            val = f.parts[-1][0]
            print(f"{k} = {val}")
        except Exception:
            pass

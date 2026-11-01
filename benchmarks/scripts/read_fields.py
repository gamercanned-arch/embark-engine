import gguf

path = r"C:\Users\abhik\Downloads\gemma-4-E4B-it-Q4_K_M.gguf"
r = gguf.GGUFReader(path)

print("=== ACTUAL METADATA VALUES ===")
for k, v in r.fields.items():
    if not k.startswith("tokenizer."):
        # decode field
        parts = v.parts
        data = v.data
        types = v.types
        # let's see how gguf reader provides value
        val = None
        if len(data) == 1:
            val = data[0]
        else:
            val = data
        print(f"{k} ({types}): {val}")

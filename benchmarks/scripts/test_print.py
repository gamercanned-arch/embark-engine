import gguf
import sys

try:
    r = gguf.GGUFReader(r"C:\Users\abhik\Downloads\Ornith-1.5-9B-Q4_K_M.gguf")
    for k in r.fields:
        if 'ssm' in k or 'block_count' in k:
            f = r.fields[k]
            # Print value
            val = [p.tolist() if hasattr(p, 'tolist') else bytes(p) for p in f.parts]
            print(k, val)
except Exception as e:
    print("Error:", e)

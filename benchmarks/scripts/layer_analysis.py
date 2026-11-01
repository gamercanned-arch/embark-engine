import gguf
from collections import defaultdict

path = r"C:\Users\abhik\Downloads\Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
reader = gguf.GGUFReader(path)

layers = defaultdict(list)
non_layers = []

for t in reader.tensors:
    if "blk." in t.name:
        l_idx = int(t.name.split(".")[1])
        layers[l_idx].append(t)
    else:
        non_layers.append(t)

print("Layer Byte Sizes and Quant Breakdown:")
print(f"{'Layer':<6} | {'Total (MB)':<10} | {'Q4_K (MB)':<10} | {'Q6_K (MB)':<10} | {'F32 (KB)':<10} | {'Type':<8} | {'V Quant':<8} | {'Down Quant':<10}")
print("-" * 80)

total_swa_bytes = 0
total_global_bytes = 0

for i in range(48):
    t_list = layers[i]
    tot_b = sum(t.n_bytes for t in t_list)
    q4_b = sum(t.n_bytes for t in t_list if t.tensor_type.name == "Q4_K")
    q6_b = sum(t.n_bytes for t in t_list if t.tensor_type.name == "Q6_K")
    f32_b = sum(t.n_bytes for t in t_list if t.tensor_type.name == "F32")
    
    is_global = (i % 6 == 5)
    l_type = "GLOBAL" if is_global else "SWA"
    if is_global:
        total_global_bytes += tot_b
    else:
        total_swa_bytes += tot_b
        
    v_t = [t for t in t_list if "attn_v" in t.name]
    v_q = v_t[0].tensor_type.name if v_t else "TIED(K)"
    
    down_t = [t for t in t_list if "ffn_down" in t.name]
    down_q = down_t[0].tensor_type.name if down_t else "NONE"
    
    print(f"{i:<6} | {tot_b/(1024**2):<10.2f} | {q4_b/(1024**2):<10.2f} | {q6_b/(1024**2):<10.2f} | {f32_b/1024:<10.2f} | {l_type:<8} | {v_q:<8} | {down_q:<10}")

print("-" * 80)
print(f"Total SWA (40 layers):    {total_swa_bytes / (1024**2):.2f} MB ({total_swa_bytes / (1024**3):.3f} GiB)")
print(f"Total Global (8 layers):  {total_global_bytes / (1024**2):.2f} MB ({total_global_bytes / (1024**3):.3f} GiB)")
print(f"Non-layer tensors:        {sum(t.n_bytes for t in non_layers) / (1024**2):.2f} MB")
print(f"Total Model Size:         {sum(t.n_bytes for t in reader.tensors) / (1024**2):.2f} MB ({sum(t.n_bytes for t in reader.tensors) / (1024**3):.3f} GiB)")

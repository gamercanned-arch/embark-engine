import json

with open("scratch/ornith_tensors.json") as f:
    tensors = json.load(f)

non_layer = [t for t in tensors if not t["name"].startswith("blk.")]
print("=== Non-layer tensors ===")
for t in non_layer:
    print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, {t['bytes']/(1024*1024):.2f} MB")

l0 = [t for t in tensors if t["name"].startswith("blk.0.")]
print("\n=== Layer 0 (GDN) ===")
for t in l0:
    print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, {t['bytes']/(1024*1024):.2f} MB")

l3 = [t for t in tensors if t["name"].startswith("blk.3.")]
print("\n=== Layer 3 (Full Attention) ===")
for t in l3:
    print(f"  {t['name']}: shape={t['shape']}, type={t['type']}, {t['bytes']/(1024*1024):.2f} MB")

# Calculate total weights per token during autoregressive decode (M=1)
# For GDN layers:
# attn_norm (4096 float = 16 KB)
# attn_qkv: Q6_K [4096, 8192] -> 26.25 MB
# attn_gate: Q4_K [4096, 4096] -> 9.00 MB
# ssm_alpha: Q4_K [4096, 32] -> 0.07 MB
# ssm_beta: Q4_K [4096, 32] -> 0.07 MB
# ssm_conv1d: F32 [4, 8192] -> 0.12 MB
# ssm_out: Q4_K [4096, 4096] -> 9.00 MB
# post_attention_norm (4096 float = 16 KB)
# ffn_gate: Q4_K [4096, 12288] -> 27.00 MB
# ffn_up: Q4_K [4096, 12288] -> 27.00 MB
# ffn_down: Q6_K [12288, 4096] -> 39.38 MB

# For Full Attention layers:
# attn_norm: 16 KB
# attn_q: Q4_K [4096, 8192] -> 18.00 MB
# attn_k: Q4_K [4096, 1024] -> 2.25 MB
# attn_v: Q6_K [4096, 1024] -> 3.28 MB
# attn_output: Q4_K [4096, 4096] -> 9.00 MB
# post_attention_norm: 16 KB
# ffn_gate: Q4_K [4096, 12288] -> 27.00 MB
# ffn_up: Q4_K [4096, 12288] -> 27.00 MB
# ffn_down: Q6_K [12288, 4096] -> 39.38 MB

total_gdn_layer_bytes = sum(t['bytes'] for t in l0)
total_attn_layer_bytes = sum(t['bytes'] for t in l3)

print(f"\nSingle GDN layer weight size: {total_gdn_layer_bytes / (1024*1024):.2f} MB")
print(f"Single Attention layer weight size: {total_attn_layer_bytes / (1024*1024):.2f} MB")

total_model_layer_bytes = 24 * total_gdn_layer_bytes + 9 * total_attn_layer_bytes
print(f"Total for 24 GDN + 9 Attention layers: {total_model_layer_bytes / (1024*1024):.2f} MB ({total_model_layer_bytes / (1024*1024*1024):.2f} GiB)")

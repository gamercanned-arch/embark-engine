import json

with open(r"c:\Users\abhik\Documents\antigravity\silly-pascal\scratch\groot_config.json") as f:
    cfg = json.load(f)

tcfg = cfg["text_config"]
h = tcfg["hidden_size"]  # 2048
layers = tcfg["num_hidden_layers"]  # 40
num_experts = tcfg["num_experts"]  # 128
top_k = tcfg["num_experts_per_tok"]  # 8
moe_inter = tcfg["moe_intermediate_size"]  # 512
shared_inter = tcfg["shared_expert_intermediate_size"]  # 512
vocab = tcfg["vocab_size"]  # 248320

# Router
router_params_per_layer = h * num_experts
total_router_params = router_params_per_layer * layers
router_bytes_fp16 = total_router_params * 2
router_bytes_fp32 = total_router_params * 4

# Experts per layer
# SwiGLU / GeGLU: gate_proj (h -> inter), up_proj (h -> inter), down_proj (inter -> h)
expert_params = 3 * (h * moe_inter)
moe_params_per_layer = num_experts * expert_params
shared_expert_params = 3 * (h * shared_inter)

active_moe_params_per_layer = top_k * expert_params + shared_expert_params

# Attention / SSM
# 30 linear_attention, 10 full_attention
linear_layers = sum(1 for x in tcfg["layer_types"] if x == "linear_attention")
full_layers = sum(1 for x in tcfg["layer_types"] if x == "full_attention")

# GDN recurrent state
# linear_num_key_heads=16, linear_key_head_dim=128 => inner_size = 2048
# linear_num_value_heads=32, linear_value_head_dim=128 => value_size = 4096
# state size per head = linear_key_head_dim * linear_value_head_dim? Or inner_size * state_size
k_dim = tcfg["linear_key_head_dim"]
v_dim = tcfg["linear_value_head_dim"]
num_k_heads = tcfg["linear_num_key_heads"]
num_v_heads = tcfg["linear_num_value_heads"]

print(f"=== GROOT 19B-A3B ARCHITECTURE BREAKDOWN ===")
print(f"Total Layers: {layers} ({linear_layers} linear_attention + {full_layers} full_attention)")
print(f"Hidden Size: {h}, Vocab: {vocab}")
print(f"Experts: {num_experts} total, {top_k} active/token + 1 shared")
print(f"Expert Intermediate Size: {moe_inter}")
print(f"Per-expert params: {expert_params:,} ({expert_params*4/1e6:.2f} MB FP32)")
print(f"MoE weights per layer: {moe_params_per_layer:,} params ({moe_params_per_layer*0.56/1e6:.2f} MB in Q4_K)")
print(f"All 40 layers MoE weights: {moe_params_per_layer*layers:,} params ({moe_params_per_layer*layers*0.56/1e9:.2f} GB in Q4_K)")
print(f"\n--- ROUTER SIZES ---")
print(f"Per-layer Router ({num_experts} x {h}): {router_params_per_layer:,} params")
print(f"  Per-layer Router size: {router_params_per_layer*2/1024:.1f} KB (FP16), {router_params_per_layer*4/1024:.1f} KB (FP32)")
print(f"Total Router across all 40 layers: {total_router_params:,} params")
print(f"  Total Router footprint: {router_bytes_fp16/(1024*1024):.2f} MB (FP16), {router_bytes_fp32/(1024*1024):.2f} MB (FP32)")

print(f"\n--- ACTIVE PARAMS PER TOKEN ---")
print(f"Active MoE per layer: {active_moe_params_per_layer:,} params ({active_moe_params_per_layer*0.56/1024/1024:.2f} MB in Q4_K)")
print(f"Total active MoE across 40 layers: {active_moe_params_per_layer*layers:,} params ({active_moe_params_per_layer*layers*0.56/1e9:.2f} GB in Q4_K)")

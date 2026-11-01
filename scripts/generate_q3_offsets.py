import gguf
import json

path = r'C:\Users\abhik\Downloads\LFM2.5-24B-A2B-UD-Q3_K_XL.gguf'
reader = gguf.GGUFReader(path)

moe_layers = {}
for t in reader.tensors:
    parts = t.name.split('.')
    if len(parts) >= 3 and parts[0] == 'blk':
        l = int(parts[1])
        t_name = '.'.join(parts[2:])
        if 'exps' in t_name or 'gate_inp' in t_name:
            if l not in moe_layers:
                moe_layers[l] = {}
            moe_layers[l][t_name] = {
                'offset': t.data_offset,
                'size': t.n_bytes,
                'type': t.tensor_type.name,
                'shape': list(t.shape)
            }

sorted_layers = sorted(moe_layers.keys())
print(f'Total MoE layers: {len(sorted_layers)}')

total_exp_bytes = 0
for l in sorted_layers:
    for k in ['ffn_down_exps.weight', 'ffn_gate_exps.weight', 'ffn_up_exps.weight']:
        total_exp_bytes += moe_layers[l][k]['size']

print(f'Total expert bytes: {total_exp_bytes} ({total_exp_bytes / (1024**3):.3f} GiB)')

# Generate C++ header
header_code = '''// Auto-generated MoE layer offsets for LFM2.5-24B-A2B-UD-Q3_K_XL.gguf
#pragma once
#include <cstdint>
#include <cstddef>

struct TensorLocation {
    uint64_t file_offset;
    size_t size_bytes;
    int type; // 0=IQ3_XXS, 1=IQ4_XS, 2=F32
};

struct MoELayerMeta {
    int layer_idx;
    TensorLocation gate_exps;
    TensorLocation up_exps;
    TensorLocation down_exps;
    TensorLocation gate_inp;
};

constexpr int TOTAL_MOE_LAYERS = 38;
constexpr uint64_t TOTAL_EXPERT_POOL_BYTES = ''' + str(total_exp_bytes) + '''ULL;

static const MoELayerMeta MOE_LAYERS_META_Q3[TOTAL_MOE_LAYERS] = {
'''

for l in sorted_layers:
    gate = moe_layers[l]['ffn_gate_exps.weight']
    up = moe_layers[l]['ffn_up_exps.weight']
    down = moe_layers[l]['ffn_down_exps.weight']
    inp = moe_layers[l]['ffn_gate_inp.weight']
    
    header_code += f"    {{ {l}, {{{gate['offset']}ULL, {gate['size']}ULL, 0}}, {{{up['offset']}ULL, {up['size']}ULL, 0}}, {{{down['offset']}ULL, {down['size']}ULL, 1}}, {{{inp['offset']}ULL, {inp['size']}ULL, 2}} }},\n"

header_code += '};\n'

out_path = r'c:\Users\abhik\Documents\antigravity\embark-engine\src\zen3_engine\moe_layer_offsets_q3.h'
with open(out_path, 'w') as f:
    f.write(header_code)

print(f'Generated {out_path}')

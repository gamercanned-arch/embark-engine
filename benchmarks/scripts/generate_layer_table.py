import gguf

reader = gguf.GGUFReader(r'C:\Users\abhik\Downloads\LFM2.5-24B-A2B-UD-IQ3_XXS.gguf')

lines = []
lines.append("// Auto-generated MoE layer offsets for LFM2.5-24B-A2B-UD-IQ3_XXS.gguf")
lines.append("#pragma once")
lines.append("#include <cstdint>")
lines.append("#include <cstddef>")
lines.append("")
lines.append("struct TensorLocation {")
lines.append("    uint64_t file_offset;")
lines.append("    size_t size_bytes;")
lines.append("    int type; // 0=IQ2_S, 1=IQ3_S, 2=IQ4_XS, 3=F32")
lines.append("};")
lines.append("")
lines.append("struct MoELayerMeta {")
lines.append("    int layer_idx;")
lines.append("    TensorLocation gate_exps;")
lines.append("    TensorLocation up_exps;")
lines.append("    TensorLocation down_exps;")
lines.append("    TensorLocation gate_inp;")
lines.append("};")
lines.append("")
lines.append("constexpr int TOTAL_MOE_LAYERS = 38;")
lines.append("static const MoELayerMeta MOE_LAYERS_META[TOTAL_MOE_LAYERS] = {")

moe_layers = {}
type_map = {'IQ2_S': 0, 'IQ3_S': 1, 'IQ4_XS': 2, 'F32': 3}

for t in reader.tensors:
    parts = t.name.split('.')
    if len(parts) >= 3 and parts[0] == 'blk':
        l = int(parts[1])
        sub = '.'.join(parts[2:])
        if sub in ['ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight', 'ffn_gate_inp.weight']:
            if l not in moe_layers:
                moe_layers[l] = {}
            # absolute file offset is already t.data_offset in gguf-py
            moe_layers[l][sub] = (int(t.data_offset), int(t.n_bytes), type_map[t.tensor_type.name])

for l in sorted(moe_layers.keys()):
    m = moe_layers[l]
    gate = m['ffn_gate_exps.weight']
    up = m['ffn_up_exps.weight']
    down = m['ffn_down_exps.weight']
    inp = m['ffn_gate_inp.weight']
    lines.append(f"    {{ {l}, {{{gate[0]}ULL, {gate[1]}ULL, {gate[2]}}}, {{{up[0]}ULL, {up[1]}ULL, {up[2]}}}, {{{down[0]}ULL, {down[1]}ULL, {down[2]}}}, {{{inp[0]}ULL, {inp[1]}ULL, {inp[2]}}} }},")

lines.append("};")

with open(r'c:\Users\abhik\Documents\antigravity\silly-pascal\scratch\moe_layer_offsets.h', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Generated moe_layer_offsets.h successfully for {len(moe_layers)} layers!")

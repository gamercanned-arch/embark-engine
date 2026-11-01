import gguf

reader = gguf.GGUFReader(r'C:\Users\abhik\Downloads\LFM2.5-24B-A2B-UD-IQ3_XXS.gguf')
print(f"Total tensors: {len(reader.tensors)}")

layers_info = {}
total_exp_bytes = 0

for t in reader.tensors:
    parts = t.name.split('.')
    if len(parts) >= 3 and parts[0] == 'blk':
        layer_idx = int(parts[1])
        t_name = '.'.join(parts[2:])
        if 'exps' in t_name or 'gate_inp' in t_name:
            if layer_idx not in layers_info:
                layers_info[layer_idx] = {}
            layers_info[layer_idx][t_name] = (t.tensor_type.name, tuple(t.shape), t.n_bytes, t.data_offset)
            if 'exps' in t_name:
                total_exp_bytes += t.n_bytes

print(f"MoE layers found: {sorted(layers_info.keys())}")
print(f"Total expert bytes: {total_exp_bytes} ({total_exp_bytes / (1024**3):.3f} GB)")

for layer in sorted(layers_info.keys()):
    info = layers_info[layer]
    gate_t = info.get('ffn_gate_exps.weight', ('-', (), 0, 0))
    up_t = info.get('ffn_up_exps.weight', ('-', (), 0, 0))
    down_t = info.get('ffn_down_exps.weight', ('-', (), 0, 0))
    inp_t = info.get('ffn_gate_inp.weight', ('-', (), 0, 0))
    print(f"L{layer:02d}: Gate={gate_t[0]} {gate_t[2]}B, Up={up_t[0]} {up_t[2]}B, Down={down_t[0]} {down_t[2]}B, Inp={inp_t[0]} {inp_t[2]}B")
